"""
Download and pre-tokenize a huggingface dataset.
Based on: https://github.com/conceptofmind/PaLM/blob/main/palm/build_dataloaders.py

Usage:
    python build_dataloaders.py --tokenizer EleutherAI/gpt-neox-20b --dataset openwebtext --text_field text --sequence_length 2048
"""
import os
import time
import json
import argparse
import multiprocessing

from loguru import logger
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer


from peft_pretraining.dataloader3 import tokenize_and_chunk


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True, help="HuggingFace tokenizer name")
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name. E.g., wikitext")
    parser.add_argument("--train_file", type=str, required=False, help="Local train dataset name. E.g., train.tsv")
    parser.add_argument("--validation_file", type=str, required=False, help="Local validation dataset name. E.g., dev.tsv")
    parser.add_argument("--test_file", type=str, required=False, help="Local test dataset name. E.g., test.tsv")
    parser.add_argument("--dataset_config", type=str, default=None, help="HuggingFace dataset config name. E.g., wikitext-2-v1")
    parser.add_argument("--text_field", type=str, default="text", help="Name of the text field in the dataset")
    parser.add_argument("--sequence_length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--num_cpu", type=int, default=multiprocessing.cpu_count(), help="Number of CPU cores")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the pre-tokenized dataset")

    parser.add_argument("--take", type=int, default=None, help="Number of examples to take from the dataset")
    args = parser.parse_args(args)

    return args


def main(args):
    print("In main")
    logger.info("*" * 40)
    logger.info(f"Starting script with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    _tokenizer_name_for_save = args.tokenizer.replace("/", "_")
    save_path = os.path.join(args.save_dir, f"{args.dataset}_{_tokenizer_name_for_save}_{args.sequence_length}")
    if args.dataset_config is not None:
        save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.dataset_config}_{_tokenizer_name_for_save}_{args.sequence_length}")

    if os.path.exists(save_path):
        raise ValueError(f"Path {save_path} already exists")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    data_files = {"train": args.train_file, "validation": args.validation_file, "test":args.test_file}
    logger.info(f"Loading the dataset in streaming mode: {args.take is not None}")
    if args.dataset is not None:
    	dataset = load_dataset(args.dataset, args.dataset_config, streaming=args.take is not None)
    elif args.train_file is not None or args.validation_file is not None:
        dataset = load_dataset("csv",data_files=data_files)
        #dataset = load_dataset("csv",data_files={"train": "combined_genome_train.csv", "validation": "combined_genome_val.csv"})
        print(dataset["train"].to_pandas())

    if args.take is not None:
        logger.info(f"Taking {args.take} examples from the dataset")
        def take(ds, n):
            return Dataset.from_generator(lambda: (yield from ds.take(n)))
        dataset_dict = {k: take(v, args.take) for k, v in dataset.items()}
        dataset = DatasetDict(dataset_dict)

    logger.info("Tokenizing and chunking the dataset")
    _time = time.time()
    dataset = tokenize_and_chunk(
        tokenizer=tokenizer,
        dataset=dataset,
        text_field=args.text_field,
        sequence_length=args.sequence_length,
        num_cpu=args.num_cpu,
    )
    print(dataset["train"].to_pandas())
    train_df = dataset["train"].to_pandas()
    train_df.to_csv("train_old.csv")
    _hours = (time.time() - _time) / 3600
    logger.info(f"Tokenization and chunking took {_hours:.2f} hours")

    dataset.save_to_disk(save_path)
    logger.info(f"Saved the dataset to {save_path}")

    with open(os.path.join(save_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    print("In main")


if __name__ == "__main__":
    print("Starting the script")
    args = parse_args()
    main(args)

