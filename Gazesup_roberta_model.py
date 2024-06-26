import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
import torch.distributed as dist
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import transformers
from transformers import RobertaTokenizer, BertConfig, AutoConfig
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaClassificationHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
	add_code_sample_docstrings,
	add_start_docstrings,
	add_start_docstrings_to_model_forward,
	replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils import model_zoo


class SP_Encoder(nn.Module):
	"""
	Head for intergrating the scanpath module.
	"""

	def __init__(self, config):
		super().__init__()
		self.sp_gen_model = Eyettention(config)
		#self.sp_gen_model.load_state_dict(torch.load('roberta_Eyettention_english.pth', map_location='cpu'))
		self.sp_gen_model.load_state_dict(model_zoo.load_url('https://github.com/aeye-lab/ACL-GazeSupervisedLM/releases/download/v1.0/roberta_Eyettention_english.pth', map_location='cpu'))

		# #freeze the parameters in scanpath generation model
		# for param in self.sp_gen_model.parameters():
		# 	param.requires_grad = False

		self.gru = nn.GRU(input_size=config.hidden_size,
							hidden_size=config.hidden_size,
							num_layers=1,
							batch_first=True,
							bidirectional=False)

		self.dropout = nn.Dropout(0.1)

	def convert_word_pos_seq_to_token_pos_seq(self,
												word_pos,
												sn_len,
												word_ids_sn
												):
		num_sent = word_pos.size(1)

		#Find the number "sn_len+1" -> the end point
		word_pos_1 = word_pos[:,0]
		sn_len_1 = sn_len[:,0]
		stop_mask_1 = (word_pos_1 == (sn_len_1+1).unsqueeze(1))
		stop_mask_1 = ~(stop_mask_1.cumsum(dim=1).cumsum(dim=1) == 1).cumsum(dim=1).bool()

		if num_sent == 2:
			word_pos_2 = word_pos[:,1]
			sn_len_2 = sn_len[:,1]
			stop_mask_2 = (word_pos_2 == (sn_len_2+1).unsqueeze(1))
			stop_mask_2 = ~(stop_mask_2.cumsum(dim=1).cumsum(dim=1) == 1).cumsum(dim=1).bool()
			SEP_indx = sn_len_1 + 1


		#compute gaze token position
		token_ids_sn = torch.arange(word_ids_sn.shape[1]).unsqueeze(0).expand(word_ids_sn.shape[0],-1).to(word_pos.device)
		word_ids_2_token_ids_sn = token_ids_sn - word_ids_sn

		gaze_token_pos = []
		for b in range(word_pos.shape[0]):
			#remove invalid predictions + SEP token
			valid_pos_seq = torch.masked_select(word_pos_1[b,:], stop_mask_1[b,:])

			if num_sent == 2:
				valid_pos_seq = torch.cat((valid_pos_seq, SEP_indx[b].reshape(1), (SEP_indx[b]+1).reshape(1))) #add two SEP token back to differentiate two sentences
				valid_pos_seq_2 = torch.masked_select(word_pos_2[b,:], stop_mask_2[b,:])[1:] + SEP_indx[b] + 1
				valid_pos_seq = torch.cat((valid_pos_seq, valid_pos_seq_2))

			try:
				assert valid_pos_seq.max() < (torch.nan_to_num(word_ids_sn[b])).max()
			except:
				import ipdb; ipdb.set_trace()

			#remove CLS token
			valid_pos_seq = valid_pos_seq[1:]
			#convert word pos sequence to token pos sequence
			cur_token_pos = torch.tensor([0.0], dtype=torch.float64).to(word_pos.device) # fake CLS token for tensor concatenation
			for p in valid_pos_seq:
				idx = torch.where(word_ids_sn[b]==p)[0]
				for i in idx:
					cur_token_pos = torch.cat((cur_token_pos, (p + word_ids_2_token_ids_sn[b][i]).reshape(1)))

			gaze_token_pos.append(cur_token_pos[1:]) # remove the fake CLS token

		sp_len = [pos.shape[0] for pos in gaze_token_pos]
		sp_len = torch.FloatTensor(sp_len).to(word_pos.device)

		#for zero length scanpath, add additional CLS token to avoid error in pack_padded_sequence operation
		for indx in torch.where(sp_len==0)[0]:
			gaze_token_pos[indx] = torch.cat((torch.zeros(1, dtype=torch.float64).to(word_pos.device), gaze_token_pos[indx]))
			sp_len[indx] = 1

		# padding. pad first seq to desired length, padding value: 511, last token index that can be retrive from BERT feature layer
		#gaze_token_pos[0] = nn.ConstantPad1d((0, 512 - gaze_token_pos[0].shape[0]), 511)(gaze_token_pos[0])

		#padding to the longest sequence, padding value: data_args.max_seq_length - 1, last token index that can be retrive from BERT feature layer
		set_max_seq_length = word_ids_sn.shape[1]
		gaze_token_pos = pad_sequence(gaze_token_pos, batch_first=True, padding_value=set_max_seq_length-1)
		return gaze_token_pos, sp_len

	def SP_Gen(self, input_ids, attention_mask, token_type_ids, word_ids, word_len, LM_word_ids):
		batch_size = input_ids.size(0)
		# Number of sentences in one instance
		# 2: pair instance;
		num_sent = input_ids.size(1)

		# Flatten input for encoding
		input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
		attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent, len)
		if token_type_ids is not None:
			token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)
		word_ids = word_ids.view((-1, word_ids.size(-1))) # (bs * num_sent, len)
		word_len = word_len.view((-1, word_len.size(-1))) # (bs * num_sent, len)

		gaze_pos, sn_len = self.sp_gen_model(sn_emd = input_ids,
												sn_mask = attention_mask,
												word_ids_sn = word_ids,
												sn_word_len = word_len,
												le = self.sp_gen_model.le)

		gaze_pos = gaze_pos.view((batch_size, num_sent, gaze_pos.size(-1))) # (bs, num_sent, hidden)
		sn_len = sn_len.view((batch_size, num_sent)) # (bs, num_sent)


		gaze_token_pos, sp_len = self.convert_word_pos_seq_to_token_pos_seq(word_pos=gaze_pos,
																			sn_len=sn_len,
																			word_ids_sn=LM_word_ids)

		return gaze_token_pos, sp_len

	def forward(self, sp_pooler_output, input_ids, attention_mask, token_type_ids, word_ids, word_len, LM_word_ids):
		gaze_token_pos, sp_len = self.SP_Gen(input_ids, attention_mask, token_type_ids, word_ids, word_len, LM_word_ids)

		#retrieve features according to scanpath ordering,
		#Note: gather can’t differentiate the index->gaze_token_pos variable
		#x_sp = torch.gather(sp_pooler_output, 1, gaze_token_pos.unsqueeze(2).repeat(1,1,768).to(torch.int64))
		#instead
		#make own one-hot encoding so that it is differentiable during training
		token_ids_sn = torch.arange(sp_pooler_output.shape[1])[None, None, :].expand(gaze_token_pos.shape[0], gaze_token_pos.shape[1], -1).to(gaze_token_pos.device)
		one_hot = token_ids_sn - gaze_token_pos.unsqueeze(-1)
		one_hot[one_hot!=0] = 1
		one_hot = 1 - one_hot

		x_sp = torch.einsum('bij,bki->bkj', sp_pooler_output, one_hot.float())
		x_sp = self.dropout(x_sp)
		x_sp_packed = pack_padded_sequence(x_sp, sp_len.cpu(), batch_first=True, enforce_sorted=False)
		x_sp_packed, last_hidden = self.gru(x_sp_packed, sp_pooler_output[:,0,:].unsqueeze(0).contiguous())

		return last_hidden[0,:]



def orig_forward(orig_self,
					encoder,
					input_ids=None,
					attention_mask=None,
					token_type_ids=None,
					position_ids=None,
					head_mask=None,
					inputs_embeds=None,
					labels=None,
					output_attentions=None,
					output_hidden_states=None,
					return_dict=None,
				):

	return_dict = return_dict if return_dict is not None else orig_self.config.use_return_dict
	batch_size = input_ids.size(0)

	# Get raw embeddings
	outputs = encoder(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=True,
			return_dict=True,
		)

	sequence_output = outputs.last_hidden_state
	logits = orig_self.classifier(sequence_output)


	loss = None
	if labels is not None:
		if orig_self.num_labels == 1:
			#  We are doing regression
			loss_fct = nn.MSELoss()
			loss = loss_fct(logits.view(-1), labels.view(-1))
		else:
			loss_fct = nn.CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, orig_self.num_labels), labels.view(-1))

	if not return_dict:
		output = (logits,) + outputs[2:]
		return ((loss,) + output) if loss is not None else output

	if loss is not None:
		return SequenceClassifierOutput(
			loss=loss,
			logits=logits,
			#hidden_states=outputs.hidden_states,
			#attentions=outputs.attentions,
		)
	else:
		return SequenceClassifierOutput(
			#loss=loss,
			logits=logits,
			#hidden_states=outputs.hidden_states,
			#attentions=outputs.attentions,
			)


def aug_forward(orig_self,
					encoder,
					input_ids=None,
					attention_mask=None,
					token_type_ids=None,
					position_ids=None,
					head_mask=None,
					inputs_embeds=None,
					labels=None,
					output_attentions=None,
					output_hidden_states=None,
					return_dict=None,
					word_ids=None,
					ET_input_ids=None,
					ET_attention_mask=None,
					ET_token_type_ids=None,
					ET_position_ids=None,
					ET_word_ids=None,
					ET_word_len=None,
				):

	return_dict = return_dict if return_dict is not None else orig_self.config.use_return_dict
	batch_size = input_ids.size(0)

	# Get raw embeddings
	outputs = encoder(
		input_ids,
		attention_mask=attention_mask,
		token_type_ids=token_type_ids,
		position_ids=position_ids,
		head_mask=head_mask,
		inputs_embeds=inputs_embeds,
		output_attentions=output_attentions,
		output_hidden_states=True,
		return_dict=True,
	) #last_hidden_state, hidden_states

	#compute L_standard
	sequence_output = outputs.last_hidden_state
	logits = orig_self.classifier(sequence_output)

	##compute L_scanpath
	sp_sequence_output = orig_self.sp_encoder(
							sp_pooler_output=sequence_output,
							input_ids=ET_input_ids,
							attention_mask=ET_attention_mask,
							token_type_ids=None,
							word_ids=ET_word_ids,
							word_len=ET_word_len,
							LM_word_ids=word_ids,
							)
	sp_logits = orig_self.classifier(sp_sequence_output.unsqueeze(1))

	if labels is not None:
		if orig_self.num_labels == 1:
			#  We are doing regression
			loss_fct = nn.MSELoss()
			loss_text = loss_fct(logits.view(-1), labels.view(-1))
			loss = loss_text + loss_fct(sp_logits.view(-1), labels.view(-1)) * orig_self.model_args.augweight
		else:
			loss_fct = nn.CrossEntropyLoss()
			loss_text = loss_fct(logits.view(-1, orig_self.num_labels), labels.view(-1))
			loss = loss_text + loss_fct(sp_logits.view(-1, orig_self.num_labels), labels.view(-1)) * orig_self.model_args.augweight

	logits = torch.cat((logits, sp_logits), dim=0)
	if not return_dict:
		output = (logits,) + outputs[2:]
		return ((loss,) + output) if loss is not None else output
	return SequenceClassifierOutput(
		loss=loss,
		logits=logits,
		#hidden_states=outputs.hidden_states,
		#attentions=outputs.attentions,
	)





class Gazesup_RobertaForSequenceClassification(RobertaPreTrainedModel):
	_keys_to_ignore_on_load_missing = [r"position_ids"]

	def __init__(self, config, *model_args, **model_kargs):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.model_args = model_kargs["model_args"]

		self.roberta = RobertaModel(config, add_pooling_layer=False)
		self.classifier = RobertaClassificationHead(config)

		self.init_weights()

	def add_sp_func(self, config):
		#for integrating the scanpath module
		self.sp_encoder = SP_Encoder(config)

	def forward(self,
		input_ids=None,
		attention_mask=None,
		token_type_ids=None,
		position_ids=None,
		head_mask=None,
		inputs_embeds=None,
		labels=None,
		output_attentions=None,
		output_hidden_states=None,
		return_dict=None,
		add_gaze=False,
		word_ids=None,
		ET_input_ids=None,
		ET_attention_mask=None,
		ET_token_type_ids=None,
		ET_position_ids=None,
		ET_word_ids=None,
		ET_word_len=None,
	):
		if self.training:
			#add gaze module
			add_gaze=True

		if add_gaze:
			return aug_forward(self, self.roberta,
				input_ids=input_ids,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids,
				position_ids=position_ids,
				head_mask=head_mask,
				inputs_embeds=inputs_embeds,
				labels=labels,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
				word_ids=word_ids,
				ET_input_ids=ET_input_ids,
				ET_attention_mask=ET_attention_mask,
				ET_token_type_ids=ET_token_type_ids,
				ET_position_ids=ET_position_ids,
				ET_word_ids=ET_word_ids,
				ET_word_len=ET_word_len,
			)
		else:
			return orig_forward(self, self.roberta,
				input_ids=input_ids,
				attention_mask=attention_mask,
				token_type_ids=token_type_ids,
				position_ids=position_ids,
				head_mask=head_mask,
				inputs_embeds=inputs_embeds,
				labels=labels,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
			)

class Eyettention(nn.Module):
	_keys_to_ignore_on_load_missing = [r"position_ids"]
	def __init__(self, config):
		super(Eyettention, self).__init__()
		self.model_pretrained = config._name_or_path
		self.used_sn_len = 24
		self.window_width = 1
		self.hidden_size = 128

		#Encode the label into interger categories, setting the exclusive category 'cf["max_sn_len"]-1' as the end sign
		self.le = LabelEncoder()
		self.le.fit(np.append(np.arange(-self.used_sn_len+3, self.used_sn_len-1), self.used_sn_len-1))
		#le.classes_

		encoder_config = AutoConfig.from_pretrained(self.model_pretrained)
		encoder_config.output_hidden_states=True
		 # initiate Bert with pre-trained weights
		print("keeping Bert with pre-trained weights")

		if 'RoBERTa' in self.model_pretrained:
			self.bert = RobertaModel.from_pretrained(self.model_pretrained, config = encoder_config, add_pooling_layer = False)
		elif 'bert' in self.model_pretrained:
			self.bert = BertModel.from_pretrained(self.model_pretrained, config = encoder_config, add_pooling_layer = False)

		self.bert.eval()
		#freeze the parameters in Bert model
		for param in self.bert.parameters():
			param.requires_grad = False

		self.embedding_dropout = nn.Dropout(0.4)
		self.encoder_lstm1 = nn.LSTM(input_size = 768, hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm2 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm3 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm4 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm5 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm6 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm7 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm8 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)

		#decoder
		self.position_embeddings = nn.Embedding(encoder_config.max_position_embeddings, encoder_config.hidden_size)
		self.LayerNorm = nn.LayerNorm(encoder_config.hidden_size, eps=encoder_config.layer_norm_eps)
		self.attn_position = nn.Linear(self.hidden_size, self.hidden_size+1) #acoount for the word length feature

		#initialize eight decoder cells
		self.decoder_cell1 = nn.LSTMCell(768, self.hidden_size)
		self.decoder_cell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell4 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell5 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell6 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell7 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell8 = nn.LSTMCell(self.hidden_size, self.hidden_size)

		#fixation postion decoder
		self.decoder_dense1 = nn.Linear(self.hidden_size*2+1, 512)
		self.decoder_dense2 = nn.Linear(512, 256)
		self.decoder_dense3 = nn.Linear(256, 256)
		self.decoder_dense4 = nn.Linear(256, 256)
		#initialize last dense layer
		self.decoder_dense5 = nn.Linear(256, self.used_sn_len*2-3)
		self.dropout_LSTM = nn.Dropout(0.2)
		self.dropout_dense = nn.Dropout(0.2)

		#for scanpath generation
		self.softmax = nn.Softmax(dim=1)

	def pool_subwords_to_word(self, subword_emb, word_ids_sn, target, pool_method='sum'):
		#try batching computing
		# Pool bert subwords back to word level
		merged_word_att = torch.empty(subword_emb.shape[0], 0, 768).to(subword_emb.device)
		if target == 'sn':
			max_len = subword_emb.size(1)

		for word_idx in range(max_len):
			word_mask = (word_ids_sn == word_idx).unsqueeze(2).repeat(1, 1, 768)
			#pooling method -> sum
			if pool_method=='sum':
				pooled_word_emb = torch.sum(subword_emb * word_mask, 1).unsqueeze(1) #[batch, 1, 768]
			elif pool_method=='mean':
				pooled_word_emb = torch.mean(subword_emb * word_mask, 1).unsqueeze(1) #[batch, 1, 768]
			merged_word_att = torch.cat([merged_word_att, pooled_word_emb], dim=1)
		mask_word = torch.sum(merged_word_att, 2).bool()
		return merged_word_att, mask_word


	def encode(self, sn_emd, sn_mask, word_ids_sn, sn_word_len):
		outputs = self.bert(input_ids=sn_emd, attention_mask=sn_mask)
		hidden_rep_orig, pooled_rep = outputs[0], outputs[1]
		# Pool bert subwords back to word level for english corpus
		merged_word_att, sn_mask_word = self.pool_subwords_to_word(hidden_rep_orig,
																	word_ids_sn,
																	target='sn',
																	pool_method='sum')

		hidden_rep = self.embedding_dropout(merged_word_att)
		#eight LSTM layers for encoder
		x, (hn, hc) = self.encoder_lstm1(hidden_rep, None)
		x, (hn, hc) = self.encoder_lstm2(self.dropout_LSTM(x), None)
		residual = x
		x, (hn, hc) = self.encoder_lstm3(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm4(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm5(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm6(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm7(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm8(self.dropout_LSTM(x), None)
		x = x + residual

		#concatenate with the word length feature
		x = torch.cat((x, sn_word_len[:, :, None].half()), dim=2)
		return x, sn_mask_word

	def location_prediction(self, sp_enc_out, word_enc_out, sp_pos, sn_mask, timestep):
		#predict fixation location
		# General Attention:
		# score(ht,hs) = (ht^T)(Wa)hs
		# hs is the output from encoder
		# ht is the previous hidden state from decoder
		# self.attn(o): [batch, step, units]
		attn_prod = torch.matmul(self.attn_position(sp_enc_out.unsqueeze(1)), word_enc_out.permute(0,2,1)) # [batch, 1, step]
		#local attention
		aligned_position = sp_pos[:, timestep]
		max_sn_len = word_enc_out.size(1)

		# Get window borders
		left = torch.where(aligned_position - self.window_width >= 0, (aligned_position - self.window_width), torch.tensor(0, dtype=torch.float).to(sn_mask.device))
		right = torch.where(aligned_position + self.window_width <= max_sn_len-1, aligned_position + self.window_width, torch.tensor(max_sn_len-1, dtype=torch.float).to(sn_mask.device))

		#exclude padding tokens
		#only consider words in the window
		sen_seq = torch.arange(max_sn_len)[None,:].expand(sn_mask.shape[0],max_sn_len).to(sn_mask.device)
		outside_win_mask = (sen_seq < left.unsqueeze(1)) +  (sen_seq > right.unsqueeze(1))
		attn_prod += (~sn_mask + outside_win_mask).unsqueeze(1) * -1e9
		att_weight = softmax(attn_prod, dim=2)             # [batch, 1, step]

		#atten_weights_batch = torch.cat([atten_weights_batch, att_weight], dim=1)
		context = torch.matmul(att_weight, word_enc_out)    # [batch, 1, units]
		hc = torch.cat([context.squeeze(1),sp_enc_out],dim=1)      # [batch, units *2]

		hc = self.dropout_dense(hc)
		hc = F.relu(self.decoder_dense1(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.decoder_dense2(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.decoder_dense3(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.decoder_dense4(hc))
		result = self.decoder_dense5(hc)                   # [batch, dec_o_dim]
		return result

	def decode(self, sn_mask, word_enc_out, sn_emd, word_ids_sn, le):
		sn_len = (torch.sum(sn_mask, axis=1)-2).float()
		# Initialize hidden state and cell state with zeros,
		hn = torch.zeros(8, sn_mask.shape[0], self.hidden_size).to(sn_mask.device)
		hc = torch.zeros(8, sn_mask.shape[0], self.hidden_size).to(sn_mask.device)
		hx, cx = hn[0,:,:], hc[0,:,:]
		hx2, cx2 = hn[1,:,:], hc[1,:,:]
		hx3, cx3 = hn[2,:,:], hc[2,:,:]
		hx4, cx4 = hn[3,:,:], hc[3,:,:]
		hx5, cx5 = hn[4,:,:], hc[4,:,:]
		hx6, cx6 = hn[5,:,:], hc[5,:,:]
		hx7, cx7 = hn[6,:,:], hc[6,:,:]
		hx8, cx8 = hn[7,:,:], hc[7,:,:]

		#use CLS token (0) as start token
		dec_in_start = (torch.zeros(sn_mask.shape[0])).long().to(sn_mask.device)
		dec_emb_in = self.bert.embeddings.word_embeddings(dec_in_start) # [batch, emb_dim]

		#add positional embeddings
		start_pos = torch.zeros(sn_mask.shape[0]).to(sn_mask.device)
		position_embeddings = self.position_embeddings(start_pos.long())
		dec_emb_in = dec_emb_in+position_embeddings
		dec_emb_in = self.LayerNorm(dec_emb_in)
		dec_in = self.embedding_dropout(dec_emb_in)

		#generate fixation one by one in an autoregressive way
		output_pos = torch.empty(sn_mask.shape[0], 0, requires_grad=True).to(sn_mask.device)
		pred_counter = 0
		output_pos = torch.cat([output_pos, start_pos.unsqueeze(1)], dim=1)
		for p in range(sn_mask.size(-1)-1):
			hx, cx = self.decoder_cell1(dec_in, (hx, cx))     # [batch, units]
			hx2, cx2 = self.decoder_cell2(self.dropout_LSTM(hx), (hx2, cx2))
			residual = hx2
			hx3, cx3 = self.decoder_cell3(self.dropout_LSTM(hx2), (hx3, cx3))
			input3 = hx3 + residual
			residual = input3
			hx4, cx4 = self.decoder_cell4(self.dropout_LSTM(input3), (hx4, cx4))
			input4 = hx4 + residual
			residual = input4
			hx5, cx5 = self.decoder_cell5(self.dropout_LSTM(input4), (hx5, cx5))
			input5 = hx5 + residual
			residual = input5
			hx6, cx6 = self.decoder_cell6(self.dropout_LSTM(input5), (hx6, cx6))
			input6 = hx6 + residual
			residual = input6
			hx7, cx7 = self.decoder_cell7(self.dropout_LSTM(input6), (hx7, cx7))
			input7 = hx7 + residual
			residual = input7
			hx8, cx8 = self.decoder_cell8(self.dropout_LSTM(input7), (hx8, cx8))
			input8 = hx8 + residual

			#location prediction
			pred_loc_logits = self.location_prediction(input8, word_enc_out, output_pos, sn_mask, p)
			if self.training:
				#Sample hard categorical using "Straight-through" trick:
				sampled_pred_loc = F.gumbel_softmax(pred_loc_logits, tau=0.5, hard=True)
			else:
				#sampling next fixation location according to the distribution
				sampled_pred_loc = torch.multinomial(self.softmax(pred_loc_logits), 1).squeeze()
				#sampled_pred_loc = pred_loc_logits.argmax(1)
				sampled_pred_loc = F.one_hot(sampled_pred_loc, num_classes=le.classes_.shape[0])

			#print(sampled_pred_loc.grad_fn)
			sac_length_class = torch.tensor(le.classes_).to(sn_mask.device).repeat(sn_mask.shape[0],1)
			sampled_sac_length = (sac_length_class * sampled_pred_loc).sum(1)
			#add saccade length -> predicted fixation word index
			pred_word_index = (output_pos[:, -1] + sampled_sac_length)

			#check the output word index for validity
			#when the prediction is end-of-sentence (23) -- set to sentence length+1, i.e. token <'SEP'>
			pred_word_index[sampled_sac_length == 23] = sn_len[sampled_sac_length == 23]+1
			#when the predicted fixation word index larger than sentence max length -- set to sentence length+1, i.e. token <'SEP'>
			pred_word_index[pred_word_index > sn_len] = sn_len[pred_word_index > sn_len]+1
			#predicted fixation word index smaller than 1 -- set to 1
			pred_word_index[pred_word_index < 1] = 1
			output_pos = torch.cat([output_pos, pred_word_index.unsqueeze(1)], dim=1)

			#prepare next timestamp input token
			pred_counter += 1
			#use predictions (token ids) as input to the next timestep
			input_ids = sn_emd * (word_ids_sn == pred_word_index.unsqueeze(1))
			mask_input_ids = ~(input_ids==0).unsqueeze(2).repeat(1,1,768)
			#merge tokens
			dec_emb_in = torch.sum(self.bert.embeddings.word_embeddings(input_ids) * mask_input_ids, axis=1)

			#add positional embeddings
			position_embeddings = self.position_embeddings(output_pos[:, -1].long())
			dec_emb_in = dec_emb_in+position_embeddings
			dec_emb_in = self.LayerNorm(dec_emb_in)
			dec_emb_in = self.embedding_dropout(dec_emb_in)

		return output_pos, sn_len                         # [batch, step, dec_o_dim]


	def forward(self, sn_emd, sn_mask, word_ids_sn, sn_word_len, le):
		x, sn_mask_word = self.encode(sn_emd, sn_mask, word_ids_sn, sn_word_len)                  # [batch, step, units], [batch, units]
		pred_pos, sn_len = self.decode(sn_mask_word, x, sn_emd, word_ids_sn, le)    # [batch, step, dec_o_dim]
		return pred_pos, sn_len


class Eyettention_pretrain(nn.Module):
	def __init__(self, cf):
		super(Eyettention_pretrain, self).__init__()
		self.cf = cf
		self.window_width = 1
		self.hidden_size = 128

		#BERT encoder
		bert_encoder_config = AutoConfig.from_pretrained(self.cf["model_pretrained"])
		bert_encoder_config.output_hidden_states=True
		 # initiate Bert with pre-trained weights
		print("keeping Bert with pre-trained weights")
		if self.cf["model_pretrained"].startswith('RoBERTa'):
			self.bert = RobertaModel.from_pretrained(self.cf["model_pretrained"], config = bert_encoder_config, add_pooling_layer = False)
		if self.cf["model_pretrained"].startswith('bert'):
			self.bert = BertModel.from_pretrained(self.cf["model_pretrained"], config = bert_encoder_config, add_pooling_layer = False)

		self.bert.eval()
		#freeze the parameters in Bert model
		for param in self.bert.parameters():
			param.requires_grad = False

		#text encoder
		self.embedding_dropout = nn.Dropout(0.4)
		self.encoder_lstm1 = nn.LSTM(input_size = 768, hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm2 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm3 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm4 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm5 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm6 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm7 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)
		self.encoder_lstm8 = nn.LSTM(input_size = int(self.hidden_size), hidden_size = int(self.hidden_size/2), num_layers = 1, batch_first=True, bidirectional=True)

		#for gaze prediction
		self.position_embeddings = nn.Embedding(bert_encoder_config.max_position_embeddings, bert_encoder_config.hidden_size)
		self.LayerNorm = nn.LayerNorm(bert_encoder_config.hidden_size, eps=bert_encoder_config.layer_norm_eps)
		self.attn_position = nn.Linear(self.hidden_size, self.hidden_size+1) #acoount for the word length feature

		#initialize eight decoder cells
		self.decoder_cell1 = nn.LSTMCell(768, self.hidden_size)
		self.decoder_cell2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell4 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell5 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell6 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell7 = nn.LSTMCell(self.hidden_size, self.hidden_size)
		self.decoder_cell8 = nn.LSTMCell(self.hidden_size, self.hidden_size)

		#fixation postion decoder
		self.decoder_dense1 = nn.Linear(self.hidden_size*2+1, 512)
		self.decoder_dense2 = nn.Linear(512, 256)
		self.decoder_dense3 = nn.Linear(256, 256)
		self.decoder_dense4 = nn.Linear(256, 256)
		#initialize last dense layer
		self.decoder_dense5 = nn.Linear(256, self.cf["max_sn_len"]*2-3)
		self.dropout_LSTM = nn.Dropout(0.2)
		self.dropout_dense = nn.Dropout(0.2)


	def pool_subwords_to_word(self, subword_emb, word_ids_sn, target, pool_method='sum'):
		#try batching computing
		# Pool bert subwords back to word level
		merged_word_att = torch.empty(subword_emb.shape[0], 0, 768).to(subword_emb.device)
		if target == 'sn':
			max_len = self.cf["max_sn_len"] #CLS and SEP included
		elif target == 'sp':
			max_len = self.cf["max_sp_len"] - 1 #do not account the 'SEP' token

		for word_idx in range(max_len):
			word_mask = (word_ids_sn == word_idx).unsqueeze(2).repeat(1, 1, 768)
			#pooling method -> sum
			if pool_method=='sum':
				pooled_word_emb = torch.sum(subword_emb * word_mask, 1).unsqueeze(1) #[batch, 1, 768]
			elif pool_method=='mean':
				pooled_word_emb = torch.mean(subword_emb * word_mask, 1).unsqueeze(1) #[batch, 1, 768]
			merged_word_att = torch.cat([merged_word_att, pooled_word_emb], dim=1)
		mask_word = torch.sum(merged_word_att, 2).bool()
		return merged_word_att, mask_word


	def encode(self, sn_emd, sn_mask, word_ids_sn, sn_word_len):
		outputs = self.bert(input_ids=sn_emd, attention_mask=sn_mask)
		hidden_rep_orig, pooled_rep = outputs[0], outputs[1]

		# Pool bert subwords back to word level for english corpus
		merged_word_att, sn_mask_word = self.pool_subwords_to_word(hidden_rep_orig,
																	word_ids_sn,
																	target='sn',
																	pool_method='sum')

		hidden_rep = self.embedding_dropout(merged_word_att)
		#eight LSTM layers for encoder
		x, (hn, hc) = self.encoder_lstm1(hidden_rep, None)
		x, (hn, hc) = self.encoder_lstm2(self.dropout_LSTM(x), None)
		residual = x
		x, (hn, hc) = self.encoder_lstm3(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm4(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm5(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm6(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm7(self.dropout_LSTM(x), None)
		x = x + residual
		residual = x
		x, (hn, hc) = self.encoder_lstm8(self.dropout_LSTM(x), None)
		x = x + residual

		#concatenate with the word length feature
		x = torch.cat((x, sn_word_len[:, :, None]), dim=2)
		return x, sn_mask_word

	def location_prediction(self, sp_enc_out, word_enc_out, sp_pos, sn_mask, timestep):
		#predict fixation location
		# General Attention:
		# score(ht,hs) = (ht^T)(Wa)hs
		# hs is the output from encoder
		# ht is the previous hidden state from decoder
		# self.attn(o): [batch, step, units]
		attn_prod = torch.matmul(self.attn_position(sp_enc_out.unsqueeze(1)), word_enc_out.permute(0,2,1)) # [batch, 1, step]
		#local attention
		aligned_position = sp_pos[:, timestep]
		# Get window borders
		left = torch.where(aligned_position - self.window_width >= 0, (aligned_position - self.window_width), 0)
		right = torch.where(aligned_position + self.window_width <= self.cf["max_sn_len"]-1, aligned_position + self.window_width, self.cf["max_sn_len"]-1)

		#exclude padding tokens
		#only consider words in the window
		sen_seq = torch.arange(self.cf["max_sn_len"])[None,:].expand(sn_mask.shape[0],self.cf["max_sn_len"]).to(sn_mask.device)
		outside_win_mask = (sen_seq < left.unsqueeze(1)) +  (sen_seq > right.unsqueeze(1))
		attn_prod += (~sn_mask + outside_win_mask).unsqueeze(1) * -1e9
		#attn_prod += (torch.Tensor.bool(1-sn_mask_word) + outside_win_mask).unsqueeze(1) * -1e9
		att_weight = softmax(attn_prod, dim=2)             # [batch, 1, step]

		#atten_weights_batch = torch.cat([atten_weights_batch, att_weight], dim=1)
		context = torch.matmul(att_weight, word_enc_out)    # [batch, 1, units]
		hc = torch.cat([context.squeeze(1),sp_enc_out],dim=1)      # [batch, units *2]

		hc = self.dropout_dense(hc)
		hc = F.relu(self.decoder_dense1(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.decoder_dense2(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.decoder_dense3(hc))
		hc = self.dropout_dense(hc)
		hc = F.relu(self.decoder_dense4(hc))
		result = self.decoder_dense5(hc)                   # [batch, dec_o_dim]
		return result

	def decode(self, sp_emd, sn_mask, sp_pos, word_enc_out, word_ids_sp):
		# Initialize hidden state and cell state with zeros,
		hn = torch.zeros(8, sp_emd.shape[0], self.hidden_size).to(sp_emd.device)
		hc = torch.zeros(8, sp_emd.shape[0], self.hidden_size).to(sp_emd.device)
		hx, cx = hn[0,:,:], hc[0,:,:]
		hx2, cx2 = hn[1,:,:], hc[1,:,:]
		hx3, cx3 = hn[2,:,:], hc[2,:,:]
		hx4, cx4 = hn[3,:,:], hc[3,:,:]
		hx5, cx5 = hn[4,:,:], hc[4,:,:]
		hx6, cx6 = hn[5,:,:], hc[5,:,:]
		hx7, cx7 = hn[6,:,:], hc[6,:,:]
		hx8, cx8 = hn[7,:,:], hc[7,:,:]

		dec_emb_in = self.bert.embeddings.word_embeddings(sp_emd[:, :-1])
		# Pool bert subwords back to word level
		sp_merged_word_emd, sp_mask_word = self.pool_subwords_to_word(dec_emb_in,
																		word_ids_sp[:,:-1],
																		target='sp',
																		pool_method='sum')

		#add positional embeddings
		position_embeddings = self.position_embeddings(sp_pos[:, :-1])
		dec_emb_in = sp_merged_word_emd+position_embeddings
		dec_emb_in = self.LayerNorm(dec_emb_in)

		dec_emb_in = dec_emb_in.permute(1,0,2)      # [step, n, emb_dim]
		dec_emb_in = self.embedding_dropout(dec_emb_in)

		#Predict output for each time step of the input features in turn
		output_pos = []
		for i in range(dec_emb_in.shape[0]):
			hx, cx = self.decoder_cell1(dec_emb_in[i], (hx, cx))     # [batch, units]
			hx2, cx2 = self.decoder_cell2(self.dropout_LSTM(hx), (hx2, cx2))
			residual = hx2
			hx3, cx3 = self.decoder_cell3(self.dropout_LSTM(hx2), (hx3, cx3))
			input3 = hx3 + residual
			residual = input3
			hx4, cx4 = self.decoder_cell4(self.dropout_LSTM(input3), (hx4, cx4))
			input4 = hx4 + residual
			residual = input4
			hx5, cx5 = self.decoder_cell5(self.dropout_LSTM(input4), (hx5, cx5))
			input5 = hx5 + residual
			residual = input5
			hx6, cx6 = self.decoder_cell6(self.dropout_LSTM(input5), (hx6, cx6))
			input6 = hx6 + residual
			residual = input6
			hx7, cx7 = self.decoder_cell7(self.dropout_LSTM(input6), (hx7, cx7))
			input7 = hx7 + residual
			residual = input7
			hx8, cx8 = self.decoder_cell8(self.dropout_LSTM(input7), (hx8, cx8))
			input8 = hx8 + residual

			pred_loc = self.location_prediction(input8, word_enc_out, sp_pos, sn_mask, i)
			output_pos.append(pred_loc)

		output_pos = torch.stack(output_pos,dim=0)                     # [step, batch, 1]
		return output_pos.permute(1,0,2)                          # [batch, step, dec_o_dim]

	def forward(self, sn_emd, sn_mask, sp_emd, sp_pos, word_ids_sn, word_ids_sp, sn_word_len):
		x, sn_mask_word = self.encode(sn_emd, sn_mask, word_ids_sn, sn_word_len)                  # [batch, step, units], [batch, units]
		pred_pos = self.decode(sp_emd, sn_mask_word, sp_pos, x, word_ids_sp)    # [batch, step, dec_o_dim]
		return pred_pos
