import itertools
import multiprocessing
from itertools import chain
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, get_worker_info
from transformers import AutoTokenizer, default_data_collator
from datasets import Dataset

from loguru import logger


class PreprocessedIterableDataset(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = itertools.islice(self.data, worker_id, None, num_workers)

        batch = []
        for i,example in enumerate(iter_data):
            tokenized_example = self.tokenizer(
                example["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            if i==0:
                print("tokenized example",tokenized_example)
            batch.append(tokenized_example)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        #input_ids = torch.stack([list(item["input_ids"]).squeeze(0) for item in batch])
        input_ids = torch.tensor([list(item["input_ids"]) for item in batch])
        #attention_mask = torch.stack([list(item["attention_mask"]).squeeze(0) for item in batch])
        attention_mask = torch.tensor([list(item["attention_mask"]) for item in batch])

        return {"input_ids": input_ids, "attention_mask": attention_mask}


def tokenize_and_chunk(
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    text_field: str,
    sequence_length: int,
    num_cpu: int = multiprocessing.cpu_count(),
):
    """
    Build data loaders for training.

    This function performs the following steps:
    1. Load the tokenizer from the pretrained "EleutherAI/gpt-neox-20b" model.
    2. Load the "openwebtext" dataset.
    3. Tokenize the dataset, adding the end-of-sentence token to each text.
    4. Process the tokenized dataset into chunks of a specified block size.

    Returns:
        Dataset: The processed dataset ready for training.
    """
    extra_map_kwargs = {"num_proc": num_cpu}  # iterable dataset does not support workers in map
    if isinstance(dataset, IterableDataset):
        extra_map_kwargs = {}

    _len_pre = len(dataset)
    # check that text_field is in dataset
    #dataset2 = dataset.map(lambda example: print([t for t in example[text_field]]))
    #dataset2 = dataset.map(lambda example: [" ".join([t[i:i+6] for i in range(0,sequence_length-6,6)]).strip(" ") for t in example[text_field]]) 
    #tokenizer.eos_token = "S"
    print(tokenizer.eos_token,type(tokenizer.eos_token))
    train_df = dataset["train"].to_pandas()
    sequence_list = ["".join(train_df[text_field].loc[0][i:i+8]) for i in range(0,sequence_length-8,4)]
    print(sequence_list+[tokenizer.eos_token])
    sequence = " ".join(sequence_list+[tokenizer.eos_token]).strip(" ")
    print(sequence)
    #lambda example: tokenizer([" ".join(["".join(example[text_field][i:i+8]) for i in range(k,k+64,8)]).strip(" ")+tokenizer.eos_token for k in range(0,sequence_length-8,64)]),
    tokenized_dataset = dataset.map(
        lambda example: tokenizer(" ".join(["".join(example[text_field][i:i+8]) for i in range(0,sequence_length-8,4)]+\
			["".join(example[text_field][i+1:i+9]) for i in range(0,sequence_length-9,4)]+\
                        ["".join(example[text_field][i+2:i+10]) for i in range(0,sequence_length-10,4)]+\
			["".join(example[text_field][i+3:i+11]) for i in range(0,sequence_length-11,4)]).strip(" ")+tokenizer.eos_token),
                        #["".join(example[text_field][i+4:i+12]) for i in range(0,sequence_length-12,4)]\
        batched=False,
        remove_columns=[text_field],
        **extra_map_kwargs,
    )
    """
    tokenized_df = tokenized_dataset["train"].to_pandas()
    padded_train_input_ids = tokenized_df["input_ids"].map(
        lambda example: [np.pad(arr, (0, max(len(arr) for arr in example) - len(arr)), mode='constant') for arr in example],
    )
    tokenized_dataset["train"].features["input_ids"] = padded_train_input_ids
    print("tokenized dataset[train].features[input_ids]",tokenized_dataset["train"].features["input_ids"])
    """
    assert "input_ids" in tokenized_dataset["train"].features
    assert len(tokenized_dataset["train"]) > 0
    logger.info(f"Tokenization finished")
    logger.info(f"\n{tokenized_dataset}")
    
    assert len(tokenized_dataset) == _len_pre
    #block_size = sequence_length//8-1
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        #concatenated_examples = {k: list(chain(*examples[k])) for k in [list(examples.keys())[0],list(examples.keys())[4],list(examples.keys())[5]]}
        concatenated_examples = {}
        
        #concatenated_examples["input_ids"] = list(chain(*examples["input_ids"]))
        concatenated_examples["input_ids"] = list(examples["input_ids"])
        #concatenated_examples["label"] = np.array(examples["label"],dtype=int)
        #print("concatenated_examples[input_ids]",concatenated_examples["input_ids"])
        total_length = len(concatenated_examples["input_ids"])
        print("total length of input ids:",total_length)
        block_size=8
        total_length = int((total_length / block_size) * block_size)
        result = {
            #k:np.array([np.array(t[i : i + block_size],dtype=int).resize((block_size,1)) for i in range(0, total_length, block_size)])
            k:np.resize(np.array(t[:total_length],dtype=int),np.load("max_len.npy").item())
            #k:np.resize(np.array(t[:total_length],dtype=int),319)
            for k, t in concatenated_examples.items()
            if k == "input_ids"
        }
        #for k, t in concatenated_examples.items():
        #    print(result[k])
        return result

        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        #if total_length >= block_size:
        # Split by chunks of max_len.
        #result["label"] = concatenated_examples["label"]
        #print(result["label"].shape,len(result["input_ids"]))
    
    t_lens = []
    for l in tokenized_dataset["train"]["input_ids"]:
        mlen = len(l)
        t_lens.append(mlen)
    max_len = np.amax(np.array(t_lens,dtype=int))
    print(max_len)
    np.save("max_len.npy",max_len)
    remove_columns = ["attention_mask"]
    train_dataset = tokenized_dataset.map(
        group_texts,
        batched=False,
        remove_columns=remove_columns,
        **extra_map_kwargs,
    )
    logger.info(f"Chunking finished")
    logger.info(f"\n{train_dataset}")
    #print(train_dataset["train"]["input_ids"])
    return train_dataset


# from https://github.com/huggingface/accelerate/blob/8514c35192ac9762920f1ab052e5cea4c0e46eeb/src/accelerate/data_loader.py#L816
class SkipBatchSampler(BatchSampler):
    """
    A `torch.utils.data.BatchSampler` that skips the first `n` batches of another `torch.utils.data.BatchSampler`.
    """

    def __init__(self, batch_sampler, skip_batches=0):
        self.batch_sampler = batch_sampler
        self.skip_batches = skip_batches

    def __iter__(self):
        for index, samples in enumerate(self.batch_sampler):
            if index >= self.skip_batches:
                yield samples

    @property
    def total_length(self):
        return len(self.batch_sampler)

    def __len__(self):
        return len(self.batch_sampler) - self.skip_batches


class SkipDataLoader(DataLoader):
    """
    Subclass of a PyTorch `DataLoader` that will skip the first batches.

    Args:
        dataset (`torch.utils.data.dataset.Dataset`):
            The dataset to use to build this datalaoder.
        skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning.
        kwargs:
            All other keyword arguments to pass to the regular `DataLoader` initialization.
    """

    def __init__(self, dataset, skip_batches=0, **kwargs):
        super().__init__(dataset, **kwargs)
        self.skip_batches = skip_batches

    def __iter__(self):
        for index, batch in enumerate(super().__iter__()):
            if index >= self.skip_batches:
                yield batch
