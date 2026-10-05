import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CharacterDataset, get_sample_data
from model import TransformerLM


def main():
    # Detect hardware accelerator
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Executing training on target device: {device}")

    # Hyperparameters tuned for fast CPU/Laptop GPU execution
    seq_len = 64
    batch_size = 32
    d_model = 128
    num_heads = 4
    num_layers = 4
    d_ff = 512
    epochs = 3
    lr = 1e-3

    raw_text = get_sample_data("input.txt")
    dataset = CharacterDataset(raw_text, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerLM(
        vocab_size=dataset.vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)

            # Reshape for CrossEntropy: (Batch * Seq_Len, Vocab_Size) vs (Batch * Seq_Len)
            loss = criterion(logits.view(-1, dataset.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} complete | Average Loss: {avg_loss:.4f}")

    print("\n" + "=" * 40)
    print("AUTOREGRESSIVE TEXT GENERATION DEMO")
    print("=" * 40)

    model.eval()
    prompt = "ROMEO:"
    prompt_tensor = torch.tensor(
        [[dataset.char_to_ix[ch] for ch in prompt]], dtype=torch.long
    ).to(device)

    out_indices = model.generate(prompt_tensor, max_new_tokens=250, temperature=0.7)
    generated_text = "".join([dataset.ix_to_char[idx.item()] for idx in out_indices[0]])
    print(generated_text)


if __name__ == "__main__":
    main()