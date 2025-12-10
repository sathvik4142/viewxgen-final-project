import torch
from dataset import ViewXGenDataset, collate_fn
from baseline.model import MiniViewXGen
from config import *

def evaluate():
    ds = ViewXGenDataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    model = MiniViewXGen().to(DEVICE)
    model.load_state_dict(torch.load("baseline.pt", map_location=DEVICE))
    model.eval()
    for batch in dl:
        inp = batch["input_tokens"].to(DEVICE)
        logits = model(inp)
        print("Eval logits:", logits.shape)
        break

if __name__ == "__main__":
    evaluate()
