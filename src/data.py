from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


dataset=load_dataset("roneneldan/TinyStories",split="train[:55000]")
dataset
tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-small")  # or "meta-llama/Llama-2-7b"
tokens = tokenizer("The cat sat on the mat.", return_tensors="pt")
train_data=dataset[:50000]
val_data=dataset[50000:55000]

vocab_size=tokenizer.vocab_size
class Tiny_dataset(Dataset):
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
        self.data = data
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
  


train_dataset=Tiny_dataset(data=train_data,tokenizer=tokenizer)
val_dataset=Tiny_dataset(data=val_data,tokenizer=tokenizer)



from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length.
    """
    # Extract 'input_ids' and 'labels' tensors from the batch
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Pad the input_ids and labels separately
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)

    return padded_input_ids, padded_labels





num_workers=2
batch_size=8

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
    batch_size=1, #for inference
    num_workers=num_workers,
    drop_last=False,
    collate_fn=collate_fn
)
print(len(val_loader))

