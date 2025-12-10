import torch
from torch.utils.data import DataLoader
from dataset import ViewXGenDataset, collate_fn
from extension.model_with_metadata import MetaViewXGen
from config import *

def train_meta():
    ds = ViewXGenDataset()
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    model = MetaViewXGen().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        total=0; count=0
        for batch in dl:
            inp = batch["input_tokens"].to(DEVICE)
            tgt = batch["target_tokens"].to(DEVICE)
            meta = torch.randint(0,16,(inp.size(0),)).to(DEVICE)

            logits = model(inp, meta)
            loss = loss_fn(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total+=loss.item(); count+=1
        print("Epoch",epoch+1,"Meta Loss",total/count)

    torch.save(model.state_dict(),"extension.pt")

if __name__ == "__main__":
    train_meta()
