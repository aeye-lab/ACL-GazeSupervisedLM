# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""
import argparse
import json
import logging
import math
import os
import sys
import random
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple

import datasets
import evaluate
import torch
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
	AutoConfig,
	AutoModelForSequenceClassification,
	AutoTokenizer,
	EvalPrediction,
	DataCollatorWithPadding,
	HfArgumentParser,
	PretrainedConfig,
	Trainer,
	TrainingArguments,
	SchedulerType,
	default_data_collator,
	get_scheduler,
	set_seed,
)
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from trainers import OurTrainer
from Gazesup_roberta_model import Gazesup_RobertaForSequenceClassification
from utils import load_feature_norm, count_parameters, remove_punctuation_split


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.27.0.dev0")
logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
	"cola": ("sentence", None),
	"mnli": ("premise", "hypothesis"),
	"mrpc": ("sentence1", "sentence2"),
	"qnli": ("question", "sentence"),
	"qqp": ("question1", "question2"),
	"rte": ("sentence1", "sentence2"),
	"sst2": ("sentence", None),
	"stsb": ("sentence1", "sentence2"),
	"wnli": ("sentence1", "sentence2"),
	"trec": ("text", None),
	"ag_news": ("text", None),
}


def load_feature_norm(path = None):
	#for Eyettention
	#load sn_word_len mean and std from the pretrained model
	if not path:
		saved_res_path = 'roberta_feature_norm_celer.pickle'
	else:
		saved_res_path = path
	file_to_read = open(saved_res_path, "rb")
	loaded_dictionary = pickle.load(file_to_read)
	sn_word_len_mean = loaded_dictionary['sn_word_len_mean']
	sn_word_len_std = loaded_dictionary['sn_word_len_std']
	return sn_word_len_mean, sn_word_len_std


def compute_word_length(txt):
	txt_word_len = [len(t) for t in txt]
	#pad nan for CLS and SEP tokens
	#txt_word_len = [np.nan] + txt_word_len + [np.nan]
	#length of a punctuation is 0, plus an epsilon to avoid division output inf
	arr = np.array(txt_word_len).astype('float64')
	arr[arr==0] = 1/(0+0.5)
	arr[arr!=0] = 1/(arr[arr!=0])
	return arr.tolist()

def pad_seq(seqs, max_len, dtype=np.compat.long, fill_value=np.nan, truncation=True):
	padded = np.full((len(seqs), max_len), fill_value=fill_value, dtype=dtype)
	for i, seq in enumerate(seqs):
		if len(seq) > max_len:
			if truncation:
				padded[i, : ] = seq[:max_len]
			else:
				print(f'Maximum sentence length larger than {max_len}, please use the flag for truncation')
				exit()
		else:
			padded[i, :len(seq)] = seq
	return padded


@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.

	Using `HfArgumentParser` we can turn this class
	into argparse arguments to be able to specify them on
	the command line.
	"""

	task_name: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
	)
	max_seq_length: int = field(
		default=128,
		metadata={
			"help": (
				"The maximum total input sequence length after tokenization. Sequences longer "
				"than this will be truncated, sequences shorter will be padded."
			)
		},
	)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
	)
	pad_to_max_length: bool = field(
		default=False,
		metadata={
			"help": (
				"Whether to pad all samples to `max_seq_length`. "
				"If False, will pad the samples dynamically when batching to the maximum length in the batch."
			)
		},
	)
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": (
				"For debugging purposes or quicker training, truncate the number of training examples to this "
				"value if set."
			)
		},
	)
	max_eval_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": (
				"For debugging purposes or quicker training, truncate the number of evaluation examples to this "
				"value if set."
			)
		},
	)
	max_predict_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": (
				"For debugging purposes or quicker training, truncate the number of prediction examples to this "
				"value if set."
			)
		},
	)
	#new
	low_resource_data_seed: Optional[int] = field(
        default=42,
        metadata={
            "help": "seed for selecting subset of the dataset if not using all."
        },
    )
	train_as_val: bool = field(
		default=True,
		metadata={"help": "if True, sample 1k from train as val"},
	)

	remove_punctuation_space: bool = field(
		default=False,
		metadata={"help": "if True, remove the space before the punctuation and return it to its original form"},
	)
	label_name: Optional[str] = field(
		default='label',
		metadata={"help": "The name of the label to use"},
	)

	def __post_init__(self):
		if self.task_name is not None:
			self.task_name = self.task_name.lower()
			if self.task_name not in task_to_keys.keys():
				raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))


@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""

	model_name_or_path: str = field(
		metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
	)
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
	)
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
	)
	cache_dir: Optional[str] = field(
		default=None,
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
	)
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
	)
	model_revision: str = field(
		default="main",
		metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
	)
	use_auth_token: bool = field(
		default=False,
		metadata={
			"help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
		},
	)
	# new arguments
	augweight: float = field(
		default=0.5,
		metadata={"help": "hyperparameter used before the gaze-integrated loss"},
	)




def main():
	# See all possible arguments in src/transformers/training_args.py
	# or by passing the --help flag to this script.
	# We now keep distinct sets of args, for a cleaner separation of concerns.

	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		# If we pass only one argument to the script and it's the path to a json file,
		# let's parse it to get our arguments.
		model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
	else:
		model_args, data_args, training_args = parser.parse_args_into_dataclasses()

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
	)

	if training_args.should_log:
		# The default of training_args.log_level is passive, so we set log level at info here to have that default.
		transformers.utils.logging.set_verbosity_info()

	log_level = training_args.get_process_log_level()
	logger.setLevel(log_level)
	datasets.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()

	# Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
		+ f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
	)
	logger.info(f"Training/evaluation parameters {training_args}")

	# Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
	# or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
	if data_args.task_name is not None:
		raw_datasets = load_dataset("glue", data_args.task_name)

	# Labels
	if data_args.task_name is not None:
		is_regression = data_args.task_name == "stsb"
		if not is_regression:
			label_list = raw_datasets["train"].features["label"].names
			num_labels = len(label_list)
		else:
			num_labels = 1

	# Set seed before initializing model.
	set_seed(training_args.seed)

	# Load pretrained model and tokenizer
	# download model & vocab.
	config = AutoConfig.from_pretrained(
		model_args.config_name if model_args.config_name else model_args.model_name_or_path,
		num_labels=num_labels,
		finetuning_task=data_args.task_name,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
	)

	tokenizer = AutoTokenizer.from_pretrained(
		model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		use_fast=model_args.use_fast_tokenizer,
		revision=model_args.model_revision,
		add_prefix_space=True #When used with is_split_into_words=True
	)

	model = Gazesup_RobertaForSequenceClassification.from_pretrained(
		model_args.model_name_or_path,
		from_tf=bool(".ckpt" in model_args.model_name_or_path),
		config=config,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
		model_args=model_args
	)

	model.add_sp_func(config)

	# Preprocessing the raw_datasets
	if data_args.task_name is not None:
		sentence1_key, sentence2_key = task_to_keys[data_args.task_name]

	#set learning rate
	task_to_lr = {'rte': 3e-5,
					'mrpc': 3e-5,
					'stsb': 2e-5,
					'sst2': 1e-5,
					'cola': 2e-5,
					'qqp': 2e-5,
					'mnli': 1e-5,
					'qnli': 2e-5,
						}
	training_args.learning_rate = task_to_lr.get(data_args.task_name)

	if data_args.task_name in ['sst2', 'mrpc']:
		data_args.remove_punctuation_space=True

	# Some models have set the order of the labels to use, so let's make sure we do use it.
	label_to_id = None
	if (
		model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
		and data_args.task_name is not None
		and not is_regression
	):
		# Some have all caps in their config, some don't.
		label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
		if sorted(label_name_to_id.keys()) == sorted(label_list):
			logger.info(
				f"The configuration of the model provided the following label correspondence: {label_name_to_id}. "
				"Using it!"
			)
			label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
		else:
			logger.info(
				"Your model seems to have been trained with labels, but they don't match the dataset: ",
				f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
				"\nIgnoring the model labels as a result.",
			)
	elif data_args.task_name is None and not is_regression:
		label_to_id = {v: i for i, v in enumerate(label_list)}

	if label_to_id is not None:
		model.config.label2id = label_to_id
		model.config.id2label = {id: label for label, id in config.label2id.items()}
	elif data_args.task_name is not None and not is_regression:
		model.config.label2id = {l: i for i, l in enumerate(label_list)}
		model.config.id2label = {id: label for label, id in config.label2id.items()}

	def preprocess_function(examples):

		total = len(examples[sentence1_key])

		features = dict()

		if sentence2_key is not None:
			num_sent = 2
			texts = examples[sentence1_key] + examples[sentence2_key]
		else:
			num_sent = 1
			texts = examples[sentence1_key]

		#The tokenizer will ignore e.g. '\ufeff' when encoding sentences.
		#The output input ids do not recognize these tokens as unknown token but just delete them,
		#but the calculated word_ids do contain it, resulting in mismatched output.
		ignore_token_by_tokenizer = ['\uf0b7', '\ufeff', '\uf105', '\uf0ba', '\uf03d', '\uf0d8', '\uf0fc', '\u202c']

		#text preprocessing
		#prepare word length inputs
		word_len_list = []
		for idx in range(int(total*num_sent)):
			if data_args.remove_punctuation_space == True:
				txt = remove_punctuation_split(texts[idx])
			else:
				txt = texts[idx]
			texts[idx] = [w for w in txt.split() if w not in ignore_token_by_tokenizer]
			text_word_len = compute_word_length(texts[idx])
			word_len_list.append(text_word_len)


		#for language model input, e.g. BERT, concatenate two sentences with SEP separator inbetween
		if sentence2_key is not None:
			texts = ((texts[:total], texts[total:]))
			word_len_list = [[word_len_list[i], word_len_list[i+total]] for i in range(total)]
		else:
			texts = ((texts,))

		text_features = tokenizer(*texts,
							padding="max_length" if data_args.pad_to_max_length else False,
							max_length=data_args.max_seq_length,
							truncation=True, #longest_first
							is_split_into_words=True)

		#Note: Roberta add two stop tokens in the middle for paried sentence, which is different from Bert
		#prepare word ids
		word_ids_ori_list = []
		word_ids_list = []
		SEP_word_idx_list = []
		for idx in range(total):
			word_ids = text_features.word_ids(idx)
			word_ids = [val if val is not None else np.nan for val in word_ids]
			word_ids_ori_list.append(word_ids.copy())
			nan_pos = np.where(np.isnan(word_ids))[0]

			if sentence2_key is not None:
				#make index start from 0, CLS -> 0 and SEP -> last index
				#Make two sentences with consecutive word IDs, so that each word id is unique
				assert nan_pos.size == 4
				word_ids[nan_pos[0]] = -1
				word_ids[nan_pos[1]] = word_ids[nan_pos[1]-1] + 1
				word_ids[nan_pos[2]] = word_ids[nan_pos[2]-1] + 1
				word_ids[nan_pos[3]] = word_ids[nan_pos[3]-1] + 1

				SEP_word_idx = nan_pos[1] #first sep position
				SEP_word_idx_list.append(SEP_word_idx)
				assert text_features['input_ids'][idx].index(2) == SEP_word_idx

				sn2_word_id = word_ids[SEP_word_idx+2 :]
				sn2_word_id = [i+2+word_ids[SEP_word_idx] for i in sn2_word_id]
				word_ids[SEP_word_idx+2 :] = sn2_word_id
				word_ids = [i+1 for i in word_ids]
				#sanity check
				try:
					assert np.array_equal(np.unique(word_ids), range(min(word_ids), max(word_ids)+1)), "unencoded tokens should be removed in advance"
				except:
					import ipdb; ipdb.set_trace()
				word_ids_list.append(word_ids)

			else:
				#make index start from 0, CLS -> 0 and SEP -> last index
				assert nan_pos.size==2 #CLS + SEP
				word_ids[nan_pos[0]] = -1
				word_ids[nan_pos[1]] = word_ids[nan_pos[1]-1] + 1
				word_ids = [i+1 for i in word_ids]

				#sanity check
				assert np.array_equal(np.unique(word_ids), range(min(word_ids), max(word_ids)+1)), "unencoded tokens should be removed in advance"
				word_ids_list.append(word_ids)

		text_features["word_ids"] = word_ids_list

		for key in text_features:
			features[key] = text_features[key]


		#for the Eyettention model input
		if sentence2_key is not None:
			#Split two consecutive sentence inputs into two single sentence inputs,
			#and we perform scanpath predictions on each sentence separately.
			features['ET_input_ids'] = [[text_features['input_ids'][idx][:SEP_word_idx_list[idx]+1], [0]+text_features['input_ids'][idx][SEP_word_idx_list[idx]+2:]] for idx in range(total)]
			#features['ET_token_type_ids'] = [[np.zeros(len(features['ET_input_ids'][idx][0]), dtype=np.int64).tolist(), np.zeros(len(features['ET_input_ids'][idx][1]), dtype=np.int64).tolist()] for idx in range(total)]
			features['ET_attention_mask'] = [[np.ones(len(features['ET_input_ids'][idx][0]), dtype=np.int64).tolist(), np.ones(len(features['ET_input_ids'][idx][1]), dtype=np.int64).tolist()] for idx in range(total)]
			features['ET_word_ids'] = [[[-1]+word_ids_ori_list[idx][1:SEP_word_idx_list[idx]]+[word_ids_ori_list[idx][SEP_word_idx_list[idx]-1]+1], [-1]+word_ids_ori_list[idx][SEP_word_idx_list[idx]+2:-1]+[word_ids_ori_list[idx][-2]+1]] for idx in range(total)]
			features['ET_word_ids'] = [[(np.array(features['ET_word_ids'][idx][0])+1).tolist(), (np.array(features['ET_word_ids'][idx][1])+1).tolist()] for idx in range(total)]
			features['ET_word_len'] = [[[np.nan]+word_len_list[idx][0][:features['ET_word_ids'][idx][0][-1]-1]+[np.nan], [np.nan]+word_len_list[idx][1][:features['ET_word_ids'][idx][1][-1]-1]+[np.nan]] for idx in range(total)]
			#sanity check
			for i in range(total):
				assert len(features['ET_input_ids'][i][0]) == len(features['ET_attention_mask'][i][0]) == len(features['ET_word_ids'][i][0])
				assert len(features['ET_word_len'][i][0]) <= len(features['ET_word_ids'][i][0])
				assert len(features['ET_input_ids'][i][1]) == len(features['ET_attention_mask'][i][1]) == len(features['ET_word_ids'][i][1])
				assert len(features['ET_word_len'][i][1]) <= len(features['ET_word_ids'][i][1])

		else:#for one single sentence, same as what we have prepared for the LM input
			features['ET_input_ids'] = [[features['input_ids'][idx]] for idx in range(total)]
			#features['ET_token_type_ids'] = [[features['token_type_ids'][idx]] for idx in range(total)]
			features['ET_attention_mask'] = [[features['attention_mask'][idx]] for idx in range(total)]
			features['ET_word_ids'] = [[features['word_ids'][idx]] for idx in range(total)]
			features['ET_word_len'] = [[[np.nan]+word_len_list[idx][:features['ET_word_ids'][idx][0][-1]-1]+[np.nan]] for idx in range(total)]

		# Map labels to IDs (not necessary for GLUE tasks)
		if "label" in examples:
			features["label"] = examples["label"]
		return features


	raw_datasets = raw_datasets.map(
		preprocess_function,
		batched=True,
		load_from_cache_file=not data_args.overwrite_cache,
		remove_columns=raw_datasets["train"].column_names,
		desc="Running tokenizer on dataset",
	)

	if training_args.do_train:
		if "train" not in raw_datasets:
			raise ValueError("--do_train requires a train dataset")
		train_dataset = raw_datasets["train"]
		if data_args.max_train_samples is not None:
			logger.warning(f'shuffling training set w. seed {data_args.low_resource_data_seed}!')
			train_dataset_all = train_dataset.shuffle(seed=data_args.low_resource_data_seed)
			train_dataset = train_dataset_all.select(range(data_args.max_train_samples))


	if training_args.do_eval:
		if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
			raise ValueError("--do_eval requires a validation dataset")
		eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
		if data_args.max_eval_samples is not None:
			eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))


	if data_args.train_as_val:
		test_dataset = eval_dataset
		eval_dataset = train_dataset_all.select(range(data_args.max_train_samples, data_args.max_train_samples + 1000))

	# Get the metric function
	if data_args.task_name is not None:
		metric = evaluate.load("glue", data_args.task_name)
	elif is_regression:
		metric = evaluate.load("mse")
	else:
		metric = evaluate.load("accuracy")

	# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
	# predictions and label_ids field) and has to return a dictionary string to float.
	def compute_metrics(p: EvalPrediction):
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
		result = metric.compute(predictions=preds, references=p.label_ids)
		if len(result) > 1:
			result["combined_score"] = np.mean(list(result.values())).item()
		return result


	# Data collator
	@dataclass
	class OurDataCollatorWithPadding:

		tokenizer: PreTrainedTokenizerBase
		padding: Union[bool, str, PaddingStrategy] = True
		max_length: Optional[int] = None
		pad_to_multiple_of: Optional[int] = None

		def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
			ET_special_keys = ['ET_input_ids', 'ET_attention_mask', 'ET_word_ids', 'ET_word_len']
			new_features = ['word_ids', 'word_len']
			bs = len(features)

			if bs > 0:
				num_sent = len(features[0]['ET_input_ids'])
			else:
				return

			#process ET features
			flat_features = []
			for feature in features: #one sample
				for i in range(num_sent):
					flat_features.append({k[3:]: feature[k][i] for k in feature if k in ET_special_keys}) #remove 'ET_' string

			origin_flat_features = []
			for feature in flat_features:
				origin_flat_features.append({k: feature[k] for k in feature if k not in new_features})

			batch = self.tokenizer.pad(
				origin_flat_features,
				padding=self.padding,
				max_length=self.max_length,
				pad_to_multiple_of=self.pad_to_multiple_of,
				return_tensors="pt",
			)

			new_features_flat_word_ids = []
			new_features_flat_word_len = []
			for feature in flat_features:
				for k in feature:
					if k == 'word_ids':
						new_features_flat_word_ids.append(feature[k])
					elif k == 'word_len':
						new_features_flat_word_len.append(feature[k])

			#Make the length of the new feature list consistent with the token sequence length to facilitate batch computation
			#truncate 'word_len' sequence to the length of token sequence or pad np.nan
			#In the most extreme case, a token is a word, and the maximum word sequence length is equal to the maximum token sequence length.
			batch_max_length = batch['input_ids'].shape[1]
			new_features_flat_word_ids = torch.tensor(pad_seq(new_features_flat_word_ids, batch_max_length, fill_value=np.nan, dtype=np.float64, truncation=True))
			new_features_flat_word_len = torch.tensor(pad_seq(new_features_flat_word_len, batch_max_length, fill_value=np.nan, dtype=np.float64, truncation=True))
			#normalize word length feature
			sn_word_len_mean, sn_word_len_std = load_feature_norm()
			new_features_flat_word_len = (new_features_flat_word_len - sn_word_len_mean)/sn_word_len_std
			new_features_flat_word_len = torch.nan_to_num(new_features_flat_word_len)
			batch['word_ids'] = new_features_flat_word_ids
			batch['word_len'] = new_features_flat_word_len

			for key in ET_special_keys:
				batch[key] = batch.pop(key[3:])


			#process LM features
			if num_sent == 2:
				special_keys = ['input_ids', 'attention_mask']
				new_features = ['word_ids']

				origin_features = []
				for feature in features:
					origin_features.append({k: feature[k] for k in feature if k in special_keys})

				batch2 = self.tokenizer.pad(
					origin_features,
					padding=self.padding,
					max_length=self.max_length,
					pad_to_multiple_of=self.pad_to_multiple_of,
					return_tensors="pt",
				)

				new_features_word_ids = []
				for feature in features:
					for k in feature:
						if k == 'word_ids':
							new_features_word_ids.append(feature[k])

				#Make the length of the new feature list consistent with the token sequence length to facilitate batch computation
				#truncate 'word_len' sequence to the length of token sequence or pad np.nan
				#In the most extreme case, a token is a word, and the maximum word sequence length is equal to the maximum token sequence length.
				batch_max_length = batch2['input_ids'].shape[1]
				new_features_word_ids = torch.tensor(pad_seq(new_features_word_ids, batch_max_length, fill_value=np.nan, dtype=np.float64, truncation=True))
				batch['word_ids'] = new_features_word_ids
				batch['input_ids'] = batch2['input_ids']
				#batch['token_type_ids'] = batch2['token_type_ids']
				batch['attention_mask'] = batch2['attention_mask']

			else:
				batch['input_ids'] = batch['ET_input_ids']
				#batch['token_type_ids'] = batch['ET_token_type_ids']
				batch['attention_mask'] = batch['ET_attention_mask']
				batch['word_ids'] = batch['ET_word_ids']

			for k in batch:
				if k in ET_special_keys:
					batch[k] = batch[k].view(bs, num_sent, -1)

			if "label" in features[0]:
				#features["label"] = examples["label"]
				label_list = []
				for k in features:
					label_list.append(k['label'])
				batch["labels"] = torch.tensor(label_list)

			return batch

	data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)

	# Initialize our Trainer
	trainer = OurTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset if training_args.do_train else None,
		eval_dataset=eval_dataset if training_args.do_eval else None,
		compute_metrics=compute_metrics,
		tokenizer=tokenizer,
		data_collator=data_collator,
	)
	trainer.model_args = model_args
	# Training
	if training_args.do_train:
		train_result = trainer.train()
		metrics = train_result.metrics
		max_train_samples = (
			data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
		)
		metrics["train_samples"] = min(max_train_samples, len(train_dataset))

		trainer.save_model()  # Saves the tokenizer too for easy upload

			# Need to save the state, since Trainer.save_model saves only the tokenizer with the model
		trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

	# Evaluation
	if training_args.do_eval:
		logger.info("*** Evaluate ***")

		# Loop to handle MNLI double evaluation (matched, mis-matched)
		tasks = [data_args.task_name]
		eval_datasets = [eval_dataset]

		for eval_dataset, task in zip(eval_datasets, tasks):
			metrics = trainer.evaluate(eval_dataset=eval_dataset)

			max_val_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
			metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

			trainer.log_metrics("eval", metrics)
			trainer.save_metrics("eval", metrics)

	if training_args.do_predict:
		logger.info("*** Test ***")
		# Loop to handle MNLI double evaluation (matched, mis-matched)
		tasks = [data_args.task_name]
		test_datasets = [test_dataset]
		# not evaluating test_mismatched
		# if data_args.task_name == "mnli":
		#     tasks.append("mnli-mm")
		#     test_datasets.append(datasets["validation_mismatched"])

		for test_dataset, task in zip(test_datasets, tasks):
			# only do_predict if train_as_val
			# Removing the `label` columns because it contains -1 and Trainer won't like that.
			#test_dataset = test_dataset.remove_columns("label")
			metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
			trainer.log_metrics("test", metrics)
			trainer.save_metrics("test", metrics)




def _mp_fn(index):
	# For xla_spawn (TPUs)
	main()


if __name__ == "__main__":
	main()
