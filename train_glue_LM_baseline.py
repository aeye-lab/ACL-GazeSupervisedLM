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
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version
from trainers import OurTrainer


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

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
		# download the dataset.
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
	)
	model = AutoModelForSequenceClassification.from_pretrained(
		model_args.model_name_or_path,
		from_tf=bool(".ckpt" in model_args.model_name_or_path),
		config=config,
		cache_dir=model_args.cache_dir,
		revision=model_args.model_revision,
	)

	# Preprocessing the raw_datasets
	if data_args.task_name is not None:
		sentence1_key, sentence2_key = task_to_keys[data_args.task_name]


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

		# Tokenize the texts
		texts = (
			(examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
		)
		result = tokenizer(*texts,
							padding="max_length" if data_args.pad_to_max_length else False,
							max_length=data_args.max_seq_length,
							truncation=True,
							)

		# Map labels to IDs (not necessary for GLUE tasks)
		if label_to_id is not None and "label" in examples:
			result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
		return result


	raw_datasets = raw_datasets.map(
		preprocess_function,
		batched=True,
		load_from_cache_file=not data_args.overwrite_cache,
		#remove_columns=raw_datasets["train"].column_names,
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

	if training_args.do_predict:
		if "test" not in raw_datasets and "test_matched" not in raw_datasets:
			raise ValueError("--do_predict requires a test dataset")
		test_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]

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

	# Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
	# we already did the padding.
	if data_args.pad_to_max_length:
		data_collator = default_data_collator
	elif training_args.fp16:
		data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
	else:
		data_collator = None

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

		#trainer.save_model()  # Saves the tokenizer too for easy upload

		output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
		if trainer.is_world_process_zero():
			with open(output_train_file, "w") as writer:
				logger.info("***** Train results *****")
				for key, value in sorted(train_result.metrics.items()):
					logger.info(f"  {key} = {value}")
					writer.write(f"{key} = {value}\n")

			# Need to save the state, since Trainer.save_model saves only the tokenizer with the model
			trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

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
			test_dataset = test_dataset.remove_columns("label")
			predictions = trainer.predict(test_dataset=test_dataset, metric_key_prefix="predict").predictions
			predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

			output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
			if trainer.is_world_process_zero():
				with open(output_test_file, "w") as writer:
					logger.info(f"***** Test results {task} *****")
					writer.write("index\tprediction\n")
					for index, item in enumerate(predictions):
						if is_regression:
							writer.write(f"{index}\t{item:3.3f}\n")
						else:
							item = label_list[item]
							writer.write(f"{index}\t{item}\n")




def _mp_fn(index):
	# For xla_spawn (TPUs)
	main()


if __name__ == "__main__":
	main()
