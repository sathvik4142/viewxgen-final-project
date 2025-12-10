import pandas as pd
import torch
from torch.utils.data import Dataset
from tokenizer_stub import ImageTokenizer
from config import DATA_CSV

VIEW_TO_ID = {"PA":0, "AP":1, "Lateral":2}

class ViewXGenDataset(Dataset):
    def __init__(self, csv_path=DATA_CSV):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = ImageTokenizer()

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = self.tokenizer.encode_image(row["image_path"])
        view_id = VIEW_TO_ID.get(str(row["view"]), 0)
        view_token = torch.tensor([view_id])

        input_tokens = torch.cat([view_token, tokens[:-1]])
        target_tokens = tokens
        return {"input_tokens": input_tokens, "target_tokens": target_tokens}

def collate_fn(batch):
    inp = torch.stack([b["input_tokens"] for b in batch])
    tgt = torch.stack([b["target_tokens"] for b in batch])
    return {"input_tokens": inp, "target_tokens": tgt}
