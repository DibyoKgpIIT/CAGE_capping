#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import re
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import json
import datasets
import evaluate
import numpy as np
from datasets import load_dataset
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    default_data_collator,
    set_seed,
)
from transformers_interpret import SequenceClassificationExplainer
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from peft_pretraining.modeling_llama import LlamaForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
import deBrujinAdjGraph as deBAG

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0")

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
}

logger = logging.getLogger(__name__)


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
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
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
        default=True,
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
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
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
    """
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    """
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

layer_outputs = {}

    
def hook_fn(module, input, output):
    layer_outputs['current'] = output.detach().cpu()



class LayerCaptureCallback(TrainerCallback):
    def __init__(self):
        self.saved_outputs = []

    def on_prediction_step(self, args=TrainingArguments, state=TrainerState, control=TrainerControl, inputs=None, outputs=None, model=None, **kwargs):
        # layer_outputs['current'] was set during the forward pass
        if 'current' in layer_outputs:
            self.saved_outputs.append(layer_outputs['current'])

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

    training_args.save_strategy = "no"

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

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
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            #use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        """
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            #use_auth_token=True if model_args.use_auth_token else None,
        )
        """
        raw_datasets = {} 
        logger.info("Loading dataset from disk")
        dataset_dict = datasets.load_from_disk(data_args.dataset_name)
        raw_datasets["train"] = dataset_dict["train"]
        raw_datasets["validation"] = dataset_dict["validation"]
        raw_datasets["test"] = dataset_dict["test"]
          
        """
        print("type(dataset_dict[train]",type(dataset_dict["train"]))
        df = dataset_dict["train"].to_pandas()
        print(df.head())  # Preview the dataset
        #train_dataset = dataset_dict["train"]
        train_scanner = dataset_dict["train"].scanner()
        train_dataset = train_scanner.to_table()
        raw_datasets["train"] = train_dataset.to_pandas()
        #if args.seed != 0:
        #    # this weird condition is due to backward compatibility
        #    train_dataset = train_dataset.shuffle(seed=args.seed)

        #eval_dataset = dataset_dict["validation"]
        validation_scanner = dataset_dict["validation"].scanner()
        eval_dataset = validation_scanner.to_table()
        raw_datasets["validation"] = eval_dataset.to_pandas()
        logger.info(f"Applying set_format")
        dataset_dict.set_format(type='torch', columns=["input_ids"])
        """
        
        # ##############################
        # Verify dataset
        logger.info("Checking datasets size")
        training_args.total_batch_size = 16
        training_args.num_training_steps = 1000000
        training_args.max_length = 512
        minimum_n_tokens = training_args.total_batch_size * training_args.num_training_steps
        dataset_n_tokens = data_args.max_train_samples * training_args.max_length
        if dataset_n_tokens < minimum_n_tokens:
            raise ValueError(f"Dataset only has {dataset_n_tokens} tokens, but we need at least {minimum_n_tokens}")

        logger.info("Loading dataset preprocessing args to check on seq_length")
        with open(os.path.join(data_args.dataset_name, "args.json")) as f:
            dataset_preprocessing_args = json.load(f)
        assert dataset_preprocessing_args["sequence_length"] == training_args.max_length
        logger.info("All good! Loading tokenizer now")
        # ##############################
        tokenizer = AutoTokenizer.from_pretrained(
            dataset_preprocessing_args["tokenizer"],
            model_max_length=training_args.max_length,
        )
        logger.info("Tokenizer loaded")

    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                #use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                #use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"

        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        #use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        #use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.model_name_or_path is None:
        model = LlamaForSequenceClassification(config)
    else:
        model = LlamaForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            #use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    hook = model.model.layers[3].post_attention_layernorm.register_forward_hook(hook_fn)
    hook2 = model.model.norm.register_forward_hook(hook_fn)
    cls_explainer = SequenceClassificationExplainer(model,tokenizer)
    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

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
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
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

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result
    """
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    """
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        #labels = model(predict_dataset)
        #print("model(predict_dataset)",labels)

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    elif is_regression:
        metric = evaluate.load("mse")
    else:
        accuracy = evaluate.load("accuracy")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")
        metric = evaluate.combine([accuracy,precision,recall,f1])

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
    callback = LayerCaptureCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[callback]
    )
    #summary(model, input_size=(batch_size, 2459))

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model(training_args.output_dir)  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    tata_regex = re.compile(r'TATA[AT][GAT][AG]')
    upe_regex = re.compile(r'[CG][CG][AG]CGCC')
    inr_regex = re.compile(r'[CT][CT]A[ATCG][AT][CT][CT]')
    dpe_regex = re.compile(r'[AG]G[AT][CT][ACG]')


    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])
        num_predicted_positive = 0
        num_predicted_negative = 0
        predicted_positive_vocab = {}
        predicted_negative_vocab = {}
        predicted_positive_vocab_full_seq = {}
        predicted_negative_vocab_full_seq = {}
        trimer_permutations = deBAG.init_all_kmer_permutation_counts('ATCG', 3)
        quadmer_permutations = deBAG.init_all_kmer_permutation_counts('ATCG', 4)
        pentamer_permutations = deBAG.init_all_kmer_permutation_counts('ATCG', 5)
        hexamer_permutations = deBAG.init_all_kmer_permutation_counts('ATCG', 6)
        heptamer_permutations = deBAG.init_all_kmer_permutation_counts('ATCG', 7)
        octamer_permutations = deBAG.init_all_kmer_permutation_counts('ATCG', 8)
        for permutation in list(trimer_permutations.keys())+list(quadmer_permutations.keys())+list(pentamer_permutations.keys())+list(hexamer_permutations.keys())+\
            list(heptamer_permutations.keys())+list(octamer_permutations.keys()):
            predicted_positive_vocab[permutation] = 0
            predicted_positive_vocab_full_seq[permutation] = 0
            predicted_negative_vocab[permutation] = 0
            predicted_negative_vocab_full_seq[permutation] = 0
        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            all_outputs = callback.saved_outputs
            print(all_outputs[0].shape)
            upe_matches_predicted1 = [0]*3
            upe_matches_predicted0 = [0]*3
            dpe_matches_predicted1 = [0]*3
            dpe_matches_predicted0 = [0]*3
            inr_matches_predicted1 = [0]*3
            inr_matches_predicted0 = [0]*3
            tata_matches_predicted1 = [0]*3
            tata_matches_predicted0 = [0]*3
            upe_matches_predicted1_union = 0
            upe_matches_predicted0_union = 0
            dpe_matches_predicted1_union = 0
            dpe_matches_predicted0_union = 0
            inr_matches_predicted1_union = 0
            inr_matches_predicted0_union = 0
            tata_matches_predicted1_union = 0
            tata_matches_predicted0_union = 0

            for i in range(len(predict_dataset)):
                avg_attention_map = all_outputs[i//8].numpy()[i%8,:,:]
                normalized_attention_map = avg_attention_map-np.min(avg_attention_map)/(np.max(avg_attention_map)-np.min(avg_attention_map))
                avg_attention_values = np.zeros((avg_attention_map.shape[0]),dtype=float)
                for row in range(avg_attention_map.shape[0]):
                    avg_attention_values[row] = np.sum(normalized_attention_map[row,:])
                attention_values_sorted = np.argsort(avg_attention_values)
                upe_matches_predicted0_flag = 0
                dpe_matches_predicted0_flag = 0
                tata_matches_predicted0_flag = 0
                inr_matches_predicted0_flag = 0
                upe_matches_predicted1_flag = 0
                dpe_matches_predicted1_flag = 0
                tata_matches_predicted1_flag = 0
                inr_matches_predicted1_flag = 0
                input_str  = tokenizer.decode(predict_dataset[i]["input_ids"])
                word_attributions = cls_explainer(str(input_str))
                predicted_class_name = cls_explainer.predicted_class_name
                if predicted_class_name=="LABEL_0":
                    num_predicted_negative+=1
                else:
                    num_predicted_positive+=1
                decoded_token_full_seq = tokenizer.decode(predict_dataset[0]["input_ids"])
                decoded_token_full_seq_no_space = decoded_token_full_seq.replace(" ","")
                decoded_token_full_seq_no_space = decoded_token_full_seq_no_space.replace("<|endoftext|>","")
                decoded_token_full_seq_no_space_trimers = deBAG.get_kmer_count_from_sequence(decoded_token_full_seq_no_space, k=3, cyclic=False)
                decoded_token_full_seq_no_space_quadmers = deBAG.get_kmer_count_from_sequence(decoded_token_full_seq_no_space, k=4, cyclic=False)
                decoded_token_full_seq_no_space_pentamers = deBAG.get_kmer_count_from_sequence(decoded_token_full_seq_no_space, k=5, cyclic=False)
                decoded_token_full_seq_no_space_hexamers = deBAG.get_kmer_count_from_sequence(decoded_token_full_seq_no_space, k=6, cyclic=False)
                decoded_token_full_seq_no_space_heptamers = deBAG.get_kmer_count_from_sequence(decoded_token_full_seq_no_space, k=7, cyclic=False)
                decoded_token_full_seq_no_space_octamers = deBAG.get_kmer_count_from_sequence(decoded_token_full_seq_no_space, k=8, cyclic=False)
                if predicted_class_name=="LABEL_0":
                    for token in decoded_token_full_seq_no_space_trimers:
                        predicted_negative_vocab_full_seq[token]+=decoded_token_full_seq_no_space_trimers[token]
                    for token in decoded_token_full_seq_no_space_quadmers:
                        predicted_negative_vocab_full_seq[token]+=decoded_token_full_seq_no_space_quadmers[token]
                    for token in decoded_token_full_seq_no_space_pentamers:
                        predicted_negative_vocab_full_seq[token]+=decoded_token_full_seq_no_space_pentamers[token]
                    for token in decoded_token_full_seq_no_space_hexamers:
                        predicted_negative_vocab_full_seq[token]+=decoded_token_full_seq_no_space_hexamers[token]
                    for token in decoded_token_full_seq_no_space_heptamers:
                        predicted_negative_vocab_full_seq[token]+=decoded_token_full_seq_no_space_heptamers[token]
                    for token in decoded_token_full_seq_no_space_octamers:
                        predicted_negative_vocab_full_seq[token]+=decoded_token_full_seq_no_space_octamers[token]
                elif predicted_class_name=="LABEL_1":
                    for token in decoded_token_full_seq_no_space_trimers:
                        predicted_positive_vocab_full_seq[token]+=decoded_token_full_seq_no_space_trimers[token]
                    for token in decoded_token_full_seq_no_space_quadmers:
                        predicted_positive_vocab_full_seq[token]+=decoded_token_full_seq_no_space_quadmers[token]
                    for token in decoded_token_full_seq_no_space_pentamers:
                        predicted_positive_vocab_full_seq[token]+=decoded_token_full_seq_no_space_pentamers[token]
                    for token in decoded_token_full_seq_no_space_hexamers:
                        predicted_positive_vocab_full_seq[token]+=decoded_token_full_seq_no_space_hexamers[token]
                    for token in decoded_token_full_seq_no_space_heptamers:
                        predicted_positive_vocab_full_seq[token]+=decoded_token_full_seq_no_space_heptamers[token]
                    for token in decoded_token_full_seq_no_space_octamers:
                        predicted_positive_vocab_full_seq[token]+=decoded_token_full_seq_no_space_octamers[token]
 

                for k in range(3):
                    decoded_token = tokenizer.decode(predict_dataset[0]["input_ids"]\
                    [attention_values_sorted[-(k+1)]-5:attention_values_sorted[-(k+1)]+5])
                    decoded_token_no_space = decoded_token.replace(" ","")
                    decoded_token_no_space = decoded_token_no_space.replace("<|endoftext|>","")
                    decoded_token_no_space_trimers = deBAG.get_kmer_count_from_sequence(decoded_token_no_space, k=3, cyclic=False)
                    decoded_token_no_space_quadmers = deBAG.get_kmer_count_from_sequence(decoded_token_no_space, k=4, cyclic=False)
                    decoded_token_no_space_pentamers = deBAG.get_kmer_count_from_sequence(decoded_token_no_space, k=5, cyclic=False)
                    decoded_token_no_space_hexamers = deBAG.get_kmer_count_from_sequence(decoded_token_no_space, k=6, cyclic=False)
                    decoded_token_no_space_heptamers = deBAG.get_kmer_count_from_sequence(decoded_token_no_space, k=7, cyclic=False)
                    decoded_token_no_space_octamers = deBAG.get_kmer_count_from_sequence(decoded_token_no_space, k=8, cyclic=False)
                    
                    print("sorted_attention_token"+str(k+1),decoded_token)
                    upe_match = upe_regex.search(decoded_token.replace(" ",""))
                    dpe_match = dpe_regex.search(decoded_token.replace(" ",""))
                    tata_match = tata_regex.search(decoded_token.replace(" ",""))
                    inr_match = inr_regex.search(decoded_token.replace(" ",""))
                    if predicted_class_name=="LABEL_0":
                        for token in decoded_token_no_space_trimers:
                            predicted_negative_vocab[token]+=decoded_token_no_space_trimers[token]
                        for token in decoded_token_no_space_quadmers:
                            predicted_negative_vocab[token]+=decoded_token_no_space_quadmers[token]
                        for token in decoded_token_no_space_pentamers:
                            predicted_negative_vocab[token]+=decoded_token_no_space_pentamers[token]
                        for token in decoded_token_no_space_hexamers:
                            predicted_negative_vocab[token]+=decoded_token_no_space_hexamers[token]
                        for token in decoded_token_no_space_heptamers:
                            predicted_negative_vocab[token]+=decoded_token_no_space_heptamers[token]
                        for token in decoded_token_no_space_octamers:
                            predicted_negative_vocab[token]+=decoded_token_no_space_octamers[token]
                        if upe_match is not None:
                            upe_matches_predicted0[k]+=1
                            upe_matches_predicted0_flag = 1
                        if dpe_match is not None:
                            dpe_matches_predicted0[k]+=1
                            dpe_matches_predicted0_flag = 1
                        if inr_match is not None:
                            inr_matches_predicted0[k]+=1
                            inr_matches_predicted0_flag = 1
                        if tata_match is not None:
                            tata_matches_predicted0[k]+=1
                            tata_matches_predicted0_flag = 1
                    else:
                        for token in decoded_token_no_space_trimers:
                            predicted_positive_vocab[token]+=decoded_token_no_space_trimers[token]
                        for token in decoded_token_no_space_quadmers:
                            predicted_positive_vocab[token]+=decoded_token_no_space_quadmers[token]
                        for token in decoded_token_no_space_pentamers:
                            predicted_positive_vocab[token]+=decoded_token_no_space_pentamers[token]
                        for token in decoded_token_no_space_hexamers:
                            predicted_positive_vocab[token]+=decoded_token_no_space_hexamers[token]
                        for token in decoded_token_no_space_heptamers:
                            predicted_positive_vocab[token]+=decoded_token_no_space_heptamers[token]
                        for token in decoded_token_no_space_octamers:
                            predicted_positive_vocab[token]+=decoded_token_no_space_octamers[token]
 
                        if upe_match is not None:
                            upe_matches_predicted1[k]+=1
                            upe_matches_predicted1_flag = 1
                        if dpe_match is not None:
                            dpe_matches_predicted1[k]+=1
                            dpe_matches_predicted1_flag = 1
                        if inr_match is not None:
                            inr_matches_predicted1[k]+=1
                            inr_matches_predicted1_flag = 1
                        if tata_match is not None:
                            tata_matches_predicted1[k]+=1
                            tata_matches_predicted1_flag = 1

                upe_matches_predicted0_union += upe_matches_predicted0_flag
                dpe_matches_predicted0_union += dpe_matches_predicted0_flag
                tata_matches_predicted0_union += inr_matches_predicted0_flag
                inr_matches_predicted0_union += tata_matches_predicted0_flag
                upe_matches_predicted1_union += upe_matches_predicted1_flag
                dpe_matches_predicted1_union += dpe_matches_predicted1_flag
                tata_matches_predicted1_union += inr_matches_predicted1_flag
                inr_matches_predicted1_union += tata_matches_predicted1_flag
            hook.remove()
            hook2.remove()
            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")
            for k in range(3):
                print("upe_matches_predicted0_"+str(k+1),upe_matches_predicted0[k]/num_predicted_negative)
                print("dpe_matches_predicted0_"+str(k+1),dpe_matches_predicted0[k]/num_predicted_negative)
                print("inr_matches_predicted0_"+str(k+1),inr_matches_predicted0[k]/num_predicted_negative)
                print("tata_matches_predicted0_"+str(k+1),tata_matches_predicted0[k]/num_predicted_negative)
                print("upe_matches_predicted1_"+str(k+1),upe_matches_predicted1[k]/num_predicted_positive)
                print("dpe_matches_predicted1_"+str(k+1),dpe_matches_predicted1[k]/num_predicted_positive)
                print("inr_matches_predicted1_"+str(k+1),inr_matches_predicted1[k]/num_predicted_positive)
                print("tata_matches_predicted1_"+str(k+1),tata_matches_predicted1[k]/num_predicted_positive)
                print("matches_predicted0_"+str(k+1),(upe_matches_predicted0[k]+dpe_matches_predicted0[k]+\
                inr_matches_predicted0[k]+tata_matches_predicted0[k])/num_predicted_negative)
                print("matches_predicted1_"+str(k+1),(upe_matches_predicted1[k]+dpe_matches_predicted1[k]+\
                inr_matches_predicted1[k]+tata_matches_predicted1[k])/num_predicted_positive)
           
            print("upe_matches_predicted0_union",upe_matches_predicted0_union/num_predicted_negative)
            print("dpe_matches_predicted0_union",dpe_matches_predicted0_union/num_predicted_negative)
            print("inr_matches_predicted0_union",inr_matches_predicted0_union/num_predicted_negative)
            print("tata_matches_predicted0_union",tata_matches_predicted0_union/num_predicted_negative)
            print("upe_matches_predicted1_union",upe_matches_predicted1_union/num_predicted_positive)
            print("dpe_matches_predicted1_union",dpe_matches_predicted1_union/num_predicted_positive)
            print("inr_matches_predicted1_union",inr_matches_predicted1_union/num_predicted_positive)
            print("tata_matches_predicted1_union",tata_matches_predicted1_union/num_predicted_positive)
            print("matches_predicted0_union",(upe_matches_predicted0_union+dpe_matches_predicted0_union+\
            inr_matches_predicted0_union+tata_matches_predicted0_union)/num_predicted_negative)
            print("matches_predicted1_union",(upe_matches_predicted1_union+dpe_matches_predicted1_union+\
            inr_matches_predicted1_union+tata_matches_predicted1_union)/num_predicted_positive)
            print("predicted_positive_vocab\n")
            for key in predicted_positive_vocab.keys():
                if predicted_positive_vocab[key]>0:
                    print(str(key)+":",predicted_positive_vocab[key])
            print("predicted_negative_vocab\n")
            for key in predicted_negative_vocab.keys():
                if predicted_negative_vocab[key]>0:
                    print(str(key)+":",predicted_negative_vocab[key])
            print("predicted_positive_vocab_full_seq\n")
            for key in predicted_positive_vocab_full_seq.keys():
                if predicted_positive_vocab_full_seq[key]>0:
                    print(str(key)+":",predicted_positive_vocab_full_seq[key])
            print("predicted_negative_vocab_full_seq\n")
            for key in predicted_negative_vocab_full_seq.keys():
                if predicted_negative_vocab_full_seq[key]>0:
                    print(str(key)+":",predicted_negative_vocab_full_seq[key])


    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    
    #print("predict_dataset[0]",predict_dataset[0])

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
