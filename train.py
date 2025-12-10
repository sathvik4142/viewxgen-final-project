import torch
from torch.utils.data import DataLoader
from dataset import ViewXGenDataset, collate_fn
from baseline.model import MiniViewXGen
from config import *

def train():
    ds = ViewXGenDataset()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model = MiniViewXGen().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        total = 0; count = 0
        for batch in dl:
            inp = batch["input_tokens"].to(DEVICE)
            tgt = batch["target_tokens"].to(DEVICE)
            logits = model(inp)
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); count += 1
        print("Epoch", epoch+1, "Loss", total/count)

    torch.save(model.state_dict(), "baseline.pt")

if __name__ == "__main__":
    train()
