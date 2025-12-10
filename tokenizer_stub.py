import torch

class ImageTokenizer:
    def __init__(self, vocab_size=256, seq_len=256):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def encode_image(self, path):
        return torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
