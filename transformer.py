# transformer.py

import time
import random
from typing import List

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt

from utils import *


class LetterCountingExample(object):
    def __init__(self, input: str, output: np.ndarray, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array(
            [vocab_index.index_of(c) for c in input],
            dtype=np.int64
        )
        self.input_tensor = torch.LongTensor(self.input_indexed)

        self.output = np.asarray(output, dtype=np.int64)
        self.output_tensor = torch.LongTensor(self.output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int = 20, batched: bool = False):
        super().__init__()
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        pos_indices = torch.arange(seq_len, dtype=torch.long, device=x.device)

        if self.batched:
            pos_emb = self.emb(pos_indices).unsqueeze(0)
            return x + pos_emb
        else:
            return x + self.emb(pos_indices)

class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, d_internal: int, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_heads = num_heads

        assert d_internal % num_heads == 0, "d_internal must be divisible by num_heads"
        self.head_dim = d_internal // num_heads

        self.W_q = nn.Linear(d_model, d_internal, bias=False)
        self.W_k = nn.Linear(d_model, d_internal, bias=False)
        self.W_v = nn.Linear(d_model, d_internal, bias=False)

        self.W_o = nn.Linear(d_internal, d_model, bias=False)

        self.ff1 = nn.Linear(d_model, d_internal)
        self.ff2 = nn.Linear(d_internal, d_model)
        self.activation = nn.ReLU()

    def forward(self, input_vecs: torch.Tensor):
        seq_len = input_vecs.size(0)

        Q = self.W_q(input_vecs)
        K = self.W_k(input_vecs)
        V = self.W_v(input_vecs)

        Q = Q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, self.d_internal)

        attn_proj = self.W_o(attn_output)
        residual_1 = input_vecs + attn_proj

        ff_hidden = self.activation(self.ff1(residual_1))
        ff_output = self.ff2(ff_hidden)
        output_vecs = residual_1 + ff_output

        attn_map = attn_weights.mean(dim=0)

        return output_vecs, attn_map


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_positions: int,
        d_model: int,
        d_internal: int,
        num_classes: int,
        num_layers: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.char_emb = nn.Embedding(vocab_size, d_model)

        self.pos_enc = PositionalEncoding(d_model, num_positions=num_positions, batched=False)

        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, d_internal, num_heads=self.num_heads) for _ in range(num_layers)]
        )

        self.out = nn.Linear(d_model, num_classes)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, indices):
        if isinstance(indices, list):
            indices = torch.LongTensor(indices)
        if indices.dim() != 1:
            indices = indices.view(-1)

        x = self.char_emb(indices)

        x = self.pos_enc(x)

        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x)
            attn_maps.append(attn)

        logits = self.out(x)
        log_probs = self.log_softmax(logits)

        return log_probs, attn_maps


def train_classifier(args, train: List[LetterCountingExample], dev: List[LetterCountingExample]):
    max_index = 0
    for ex in train:
        max_index = max(max_index, int(ex.input_tensor.max().item()))
    vocab_size = max_index + 1

    num_positions = len(train[0].input)

    d_model = 64
    d_internal = 64
    num_classes = 3
    num_layers = 1
    num_heads = 1

    model = Transformer(
        vocab_size=vocab_size,
        num_positions=num_positions,
        d_model=d_model,
        d_internal=d_internal,
        num_classes=num_classes,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    loss_fcn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        loss_this_epoch = 0.0

        ex_idxs = list(range(len(train)))
        random.seed(epoch)
        random.shuffle(ex_idxs)

        for idx in ex_idxs:
            ex = train[idx]
            inp = ex.input_tensor
            target = ex.output_tensor

            log_probs, _ = model(inp)

            loss = loss_fcn(log_probs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_this_epoch += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs} - training loss: {loss_this_epoch:.4f}")

    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################

def decode(model: Transformer,
           dev_examples: List[LetterCountingExample],
           do_print: bool = False,
           do_plot_attn: bool = False):

    num_correct = 0
    num_total = 0

    if len(dev_examples) > 100:
        print(f"Decoding on a large number of examples ({len(dev_examples)}); "
              f"not printing or plotting")
        do_print = False
        do_plot_attn = False

    for i, ex in enumerate(dev_examples):
        log_probs, attn_maps = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().cpu().numpy(), axis=1)

        if do_print:
            print(f"INPUT {i}: {ex.input}")
            print(f"GOLD  {i}: {repr(ex.output.astype(int))}")
            print(f"PRED  {i}: {repr(predictions)}")

        if do_plot_attn:
            for j, attn_map in enumerate(attn_maps):
                fig, ax = plt.subplots()
                attn_np = attn_map.detach().cpu().numpy()
                im = ax.imshow(attn_np, cmap="hot", interpolation="nearest")

                ax.set_xticks(np.arange(len(ex.input)))
                ax.set_yticks(np.arange(len(ex.input)))
                ax.set_xticklabels(list(ex.input))
                ax.set_yticklabels(list(ex.input))
                ax.xaxis.tick_top()

                plt.tight_layout()
                plt.savefig(f"plots/{i}_attns{j}.png")
                plt.close(fig)

        acc = sum(
            int(predictions[t] == ex.output[t])
            for t in range(len(predictions))
        )
        num_correct += acc
        num_total += len(predictions)

    accuracy = float(num_correct) / float(num_total) if num_total > 0 else 0.0
    print(f"Accuracy: {num_correct} / {num_total} = {accuracy:.6f}")
