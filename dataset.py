import os
import ssl
import urllib.request
import torch
from torch.utils.data import Dataset


class CharacterDataset(Dataset):
    """Character-level language modeling dataset."""

    def __init__(self, text: str, seq_len: int):
        self.seq_len = seq_len
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.data = torch.tensor([self.char_to_ix[ch] for ch in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def get_sample_data(data_path: str = "input.txt") -> str:
    """Utility to load or download dataset robustly."""
    if not os.path.exists(data_path):
        print("Downloading Tiny Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

        # Create unverified SSL context to prevent Windows SSL certificate errors
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, context=ctx) as response:
            text_data = response.read().decode("utf-8")

        with open(data_path, "w", encoding="utf-8") as f:
            f.write(text_data)

    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text