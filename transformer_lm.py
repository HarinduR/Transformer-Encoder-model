# models.py

import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from transformer import PositionalEncoding


class LanguageModel(object):
    def get_next_char_log_probs(self, context: str) -> np.ndarray:
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars: str, context: str) -> float:
        raise Exception("Only implemented in subclasses")

class UniformLanguageModel(LanguageModel):
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self._log_uniform = math.log(1.0 / float(vocab_size))

    def get_next_char_log_probs(self, context: str) -> np.ndarray:
        return np.ones(self.vocab_size, dtype=np.float64) * self._log_uniform

    def get_log_prob_sequence(self, next_chars: str, context: str) -> float:
        return self._log_uniform * len(next_chars)

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.char_emb = nn.Embedding(vocab_size, d_model)

        self.pos_enc = PositionalEncoding(
            d_model=d_model,
            num_positions=max_seq_len,
            batched=True,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.out = nn.Linear(d_model, vocab_size)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, input_indices: torch.LongTensor) -> torch.Tensor:
        if input_indices.dim() == 1:
            input_indices = input_indices.unsqueeze(0)

        batch_size, seq_len = input_indices.shape
        device = input_indices.device

        x = self.char_emb(input_indices) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        attn_mask = self._generate_causal_mask(seq_len, device=device)

        x = self.encoder(x, mask=attn_mask)

        logits = self.out(x)
        log_probs = self.log_softmax(logits)

        return log_probs

class NeuralLanguageModel(LanguageModel):
    def __init__(self, model: TransformerLM, vocab_index):
        self.model = model
        self.vocab_index = vocab_index
        self.vocab_size = len(vocab_index)
        self.device = next(model.parameters()).device

    def _string_to_indices_tensor(self, text: str) -> torch.LongTensor:
        indices: List[int] = []
        for c in text:
            idx = self.vocab_index.index_of(c)
            if idx == -1:
                idx = self.vocab_index.index_of(" ")
            indices.append(idx)

        if not indices:
            idx = self.vocab_index.index_of(" ")
            indices = [idx]

        max_len = self.model.max_seq_len
        if len(indices) > max_len:
            indices = indices[-max_len:]

        indices_tensor = torch.LongTensor(indices).to(self.device)
        return indices_tensor

    def get_next_char_log_probs(self, context: str) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            context_indices = self._string_to_indices_tensor(context)
            log_probs = self.model(context_indices)
            next_log_probs = log_probs[0, -1, :].detach().cpu().numpy()
        return next_log_probs

    def get_log_prob_sequence(self, next_chars: str, context: str) -> float:
        total_log_prob = 0.0
        running_context = context

        for c in next_chars:
            log_probs = self.get_next_char_log_probs(running_context)
            idx = self.vocab_index.index_of(c)
            if idx == -1:
                idx = self.vocab_index.index_of(" ")
            total_log_prob += float(log_probs[idx])
            running_context += c

        return total_log_prob

def _make_chunks(indices: np.ndarray, seq_len: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
    N = len(indices)
    inputs = []
    targets = []

    max_start = N - (seq_len + 1)
    for s in range(0, max_start + 1, seq_len):
        window = indices[s: s + seq_len + 1]
        inp = window[:-1]
        tgt = window[1:]
        inputs.append(inp)
        targets.append(tgt)

    inputs = torch.LongTensor(np.stack(inputs, axis=0))
    targets = torch.LongTensor(np.stack(targets, axis=0))
    return inputs, targets


def train_lm(args, train_text: str, dev_text: str, vocab_index) -> LanguageModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_indices = np.array(
        [vocab_index.index_of(c) for c in train_text],
        dtype=np.int64,
    )
    dev_indices = np.array(
        [vocab_index.index_of(c) for c in dev_text],
        dtype=np.int64,
    )

    vocab_size = len(vocab_index)

    seq_len = 64
    d_model = 256
    n_heads = 4
    num_layers = 3
    dim_ff = 512
    dropout = 0.1
    batch_size = 64
    num_epochs = 25
    lr = 1e-3

    train_inputs, train_targets = _make_chunks(train_indices, seq_len)
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    print(f"Number of training chunks: {len(train_dataset)}")
    print(f"Sequence length per chunk: {seq_len}")

    core_model = TransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        max_seq_len=seq_len,
        dropout=dropout,
    ).to(device)

    loss_fcn = nn.NLLLoss()
    optimizer = optim.Adam(core_model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        core_model.train()
        total_loss = 0.0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            log_probs = core_model(batch_inputs)

            B, T, V = log_probs.shape
            loss = loss_fcn(
                log_probs.view(B * T, V),
                batch_targets.view(B * T),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs} - training loss: {avg_loss:.4f}")

        core_model.eval()
        with torch.no_grad():
            dev_ids = torch.LongTensor(dev_indices).to(device)
            if len(dev_ids) > seq_len + 1:
                dev_window = dev_ids[:seq_len + 1]
            else:
                dev_window = dev_ids

            dev_inp = dev_window[:-1].unsqueeze(0)
            dev_tgt = dev_window[1:].unsqueeze(0)

            dev_log_probs = core_model(dev_inp)
            B, T, V = dev_log_probs.shape
            dev_loss = loss_fcn(
                dev_log_probs.view(B * T, V),
                dev_tgt.view(B * T),
            )
            dev_perplexity = math.exp(dev_loss.item())
            print(f"  Approx dev perplexity (prefix): {dev_perplexity:.4f}")

    core_model.eval()
    lm = NeuralLanguageModel(core_model, vocab_index)
    return lm
