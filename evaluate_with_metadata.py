import torch
from dataset import ViewXGenDataset, collate_fn
from extension.model_with_metadata import MetaViewXGen
from config import *

def evaluate_meta():
    ds = ViewXGenDataset()
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    model = MetaViewXGen().to(DEVICE)
    model.load_state_dict(torch.load("extension.pt", map_location=DEVICE))
    model.eval()
    for batch in dl:
        inp = batch["input_tokens"].to(DEVICE)
        meta = torch.randint(0,16,(inp.size(0),)).to(DEVICE)
        logits = model(inp, meta)
        print("Eval logits:", logits.shape)
        break

if __name__ == "__main__":
    evaluate_meta()
