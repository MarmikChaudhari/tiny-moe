from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset as torchDataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import os

n_samples_simple_stories = 10000 # 1_000_000 
n_samples_code = 5000 # 210_000 
n_samples_arxiv = 1000 # 21_000 # down sample as avg sample length of arxiv is way higher than other subsets
test_split = 0.05
num_workers = 4
batch_size = 256 # 512/ 256
vocab_size   = None
tokenizer    = None
train_loader = None
val_loader   = None

# dataset=load_dataset("roneneldan/TinyStories",split="train[:55000]")
# train_data=dataset[:50000]
# val_data=dataset[50000:55000]


def get_dataset_tokenizer(n_samples_simple_stories=n_samples_simple_stories,n_samples_code=n_samples_code,n_samples_arxiv=n_samples_arxiv,test_split=test_split):
    # get the three subsets of arxiv, code and simple stories
    arxiv_url = [
        f"https://olmo-data.org/dolma-v1_7/redpajama-arxiv/arxiv-{i:04d}.json.gz" for i in range(2)
    ]
    code_url = ["https://olmo-data.org/dolma-v1_7/starcoder/starcoder-0000.json.gz"]
    simeple_stories_ds = load_dataset("SimpleStories/SimpleStories", split="train", streaming=True)

    arxiv_ds = load_dataset(
        "json",
        data_files=arxiv_url,
        split="train",
        streaming=True,
    )
    code_ds = load_dataset(
        "json",
        data_files=code_url,
        split="train",
        streaming=True,
    )

    # Manual iteration with progress bars
    simple_stories_data = []
    for i, item in enumerate(tqdm(simeple_stories_ds, desc="SimpleStories", total=n_samples_simple_stories)):
        if i >= n_samples_simple_stories:
            break
        simple_stories_data.append({"text": item["story"]})
    
    arxiv_data = []
    for i, item in enumerate(tqdm(arxiv_ds, desc="Arxiv", total=n_samples_arxiv)):
        if i >= n_samples_arxiv:
            break
        arxiv_data.append({"text": item["text"]})
    
    code_data = []
    for i, item in enumerate(tqdm(code_ds, desc="Code", total=n_samples_code)):
        if i >= n_samples_code:
            break
        code_data.append({"text": item["text"]})

    datasets = [
        Dataset.from_list(simple_stories_data),
        Dataset.from_list(arxiv_data),
        Dataset.from_list(code_data)
    ]

    # combine
    combined_ds = concatenate_datasets(datasets)

    print(f"total items: {len(combined_ds)}")

    # train/val split
    train_test_split = combined_ds.train_test_split(test_size=test_split, seed=42)
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']

    # convert to dictionary format with "text" key
    train_data = {"text": train_dataset["text"]} # the value of this dict is a list of the text samples
    val_data = {"text": val_dataset["text"]}

    # print(f"train items: {len(train_data['text'])}")
    # print(f"val items: {len(val_data['text'])}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2", legacy=False)  # or "meta-llama/Llama-2-7b", use gpt2 tokenizer (50k vocab size basically)
    # set pad_token to eos_token since GPT-2 doesn't have a dedicated pad token
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size=tokenizer.vocab_size

    return train_data, val_data, tokenizer, vocab_size


class Tiny_dataset(torchDataset):
    def __init__(self, data, tokenizer, max_seq_length=150):
        """
        Initializes a TinyDataset instance.

        Args:
            data (dict): A dictionary containing the dataset with a "text" key.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to encode the text data.
            max_seq_length (int, optional): The maximum sequence length for tokenization. Defaults to 150.

        Attributes:
            data (dict): The dataset with text data.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for encoding.
            encoded_texts (list): A list of encoded text sequences.
            max_seq_length (int): The maximum sequence length for tokenization.
        """
        self.dataset = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Encode text data and store as encoded texts
        self.encoded_texts = [
            tokenizer.encode(
                text,
                truncation=True,
                max_length=max_seq_length + 1,  # Add 1 to handle the shifting of labels
                padding=False
            ) for text in tqdm(data["text"], "Encoding")
        ]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        
        # Create input_ids and labels
        input_ids = torch.tensor(encoded[:-1], dtype=torch.long)  # Exclude last token for input
        labels = torch.tensor(encoded[1:], dtype=torch.long)  # Exclude first token for labels

        return {
            "input_ids": input_ids,
            "labels": labels
        }

    def __len__(self):
        return len(self.encoded_texts)

# pad_token_id = None

def init_dataset(n_samples_simple_stories=n_samples_simple_stories,n_samples_code=n_samples_code,n_samples_arxiv=n_samples_arxiv,test_split=test_split):
    
    # Check if processed tokens exist
    if os.path.exists("processed_data/train_tokenized") and os.path.exists("processed_data/val_tokenized"):
        print("Loading processed tokens from disk...")
        
        # Load tokenized datasets
        train_tokenized_hf = load_from_disk("processed_data/train_tokenized")
        val_tokenized_hf = load_from_disk("processed_data/val_tokenized")
        
        # Get tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2", legacy=False)
        tokenizer.pad_token = tokenizer.eos_token
        vocab_size = tokenizer.vocab_size
        
        # Create custom dataset class that directly uses the tokenized data
        class PreTokenizedDataset(torchDataset):
            def __init__(self, tokenized_data):
                self.input_ids = tokenized_data["input_ids"]
                self.labels = tokenized_data["labels"]
            
            def __getitem__(self, index):
                return {
                    "input_ids": torch.tensor(self.input_ids[index], dtype=torch.long),
                    "labels": torch.tensor(self.labels[index], dtype=torch.long)
                }
            
            def __len__(self):
                return len(self.input_ids)
        
        train_dataset = PreTokenizedDataset(train_tokenized_hf)
        val_dataset = PreTokenizedDataset(val_tokenized_hf)
        
        print("Processed tokens loaded successfully!")
        return train_dataset, val_dataset, tokenizer, vocab_size
    
    else:
        print("No processed tokens found. Creating new datasets...")
        train_data, val_data, tokenizer, vocab_size = get_dataset_tokenizer(n_samples_simple_stories, n_samples_code, n_samples_arxiv, test_split)
        train_dataset=Tiny_dataset(data=train_data,tokenizer=tokenizer)
        val_dataset=Tiny_dataset(data=val_data,tokenizer=tokenizer)
        
        # save the tokenized datasets to disk
        print("Saving tokenized datasets to disk...")
        os.makedirs("processed_data", exist_ok=True)

        # convert the torch datasets back to HuggingFace format for saving
        train_tokenized_data = {
            "input_ids": [item["input_ids"].tolist() for item in train_dataset],
            "labels": [item["labels"].tolist() for item in train_dataset]
        }

        val_tokenized_data = {
            "input_ids": [item["input_ids"].tolist() for item in val_dataset],
            "labels": [item["labels"].tolist() for item in val_dataset]
        }

        # create HuggingFace datasets from the tokenized data
        train_tokenized_hf = Dataset.from_dict(train_tokenized_data)
        val_tokenized_hf = Dataset.from_dict(val_tokenized_data)

        # save to disk
        train_tokenized_hf.save_to_disk("processed_data/train_tokenized")
        val_tokenized_hf.save_to_disk("processed_data/val_tokenized")

        print("tokenized datasets saved to processed_data/train_tokenized and processed_data/val_tokenized")
        return train_dataset, val_dataset, tokenizer, vocab_size


class CollateFunction:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        """
        custom collate function to pad sequences to the same length.
        """
        # Extract 'input_ids' and 'labels' tensors from the batch
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        # Pad the input_ids and labels separately
        padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=self.pad_token_id)

        return padded_input_ids, padded_labels
    

if __name__ == "__main__":
    # only runs when script executed directly
    train_dataset, val_dataset, tokenizer, vocab_size = init_dataset(n_samples_simple_stories=n_samples_simple_stories,n_samples_code=n_samples_code,n_samples_arxiv=n_samples_arxiv,test_split=test_split)
    collate_fn = CollateFunction(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size, #for inference
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn
        )

    print(f"number of items in val_loader: {len(val_loader)}")

else:
    import multiprocessing
    if multiprocessing.current_process().name == 'MainProcess':
        # only main process creates datasets
        train_dataset, val_dataset, tokenizer, vocab_size = init_dataset(n_samples_simple_stories=n_samples_simple_stories,n_samples_code=n_samples_code,n_samples_arxiv=n_samples_arxiv,test_split=test_split)
        collate_fn = CollateFunction(pad_token_id=tokenizer.pad_token_id)
        # create data loaders
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate_fn
        )
        print(f"number of items in val_loader: {len(val_loader)}")