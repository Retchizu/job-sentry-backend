"""HybridFusedClassifier and LSTM-side preprocessing (matches phase6_hybrid_fused.ipynb §6–§7)."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch
import torch.nn as nn
from transformers import DistilBertModel


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower())


def _text_to_lstm_ids(
    text: str,
    word2idx: dict[str, int],
    max_len: int,
    oov_idx: int = 1,
    pad_idx: int = 0,
) -> list[int]:
    tokens = tokenize_words(text)
    ids = [word2idx.get(w, oov_idx) for w in tokens]
    ids = ids[:max_len]
    ids = ids + [pad_idx] * (max_len - len(ids))
    return ids


def texts_to_lstm_batch(
    texts: Sequence[str],
    word2idx: dict[str, int],
    max_len: int,
    oov_idx: int = 1,
    pad_idx: int = 0,
) -> torch.Tensor:
    rows = [_text_to_lstm_ids(t, word2idx, max_len, oov_idx, pad_idx) for t in texts]
    return torch.tensor(rows, dtype=torch.long)


class HybridFusedClassifier(nn.Module):
    """
    Single end-to-end model fusing DistilBERT and BiLSTM for binary classification.

    Forward inputs
    --------------
    input_ids      : LongTensor [B, MAX_LEN_BERT]   — DistilBERT subword IDs
    attention_mask : LongTensor [B, MAX_LEN_BERT]   — DistilBERT attention mask
    lstm_ids       : LongTensor [B, MAX_LEN_BILSTM] — word-level vocab IDs (padded)

    Forward output
    --------------
    logits : FloatTensor [B, NUM_LABELS=2]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        lstm_hidden: int = 64,
        fusion_hidden: int = 256,
        num_labels: int = 2,
        dropout: float = 0.3,
        distilbert_name: str = "distilbert-base-uncased",
    ):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(distilbert_name)
        bert_dim = self.bert.config.dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        lstm_out_dim = lstm_hidden * 2

        fused_dim = bert_dim + lstm_out_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_labels),
        )

    def freeze_bert(self) -> None:
        for p in self.bert.parameters():
            p.requires_grad = False

    def unfreeze_bert(self) -> None:
        for p in self.bert.parameters():
            p.requires_grad = True

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lstm_ids: torch.Tensor,
    ) -> torch.Tensor:
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_vec = self._mean_pool(bert_out.last_hidden_state, attention_mask)

        embedded = self.embedding(lstm_ids)
        _, (h_n, _) = self.lstm(embedded)
        lstm_vec = torch.cat([h_n[0], h_n[1]], dim=-1)

        fused = torch.cat([bert_vec, lstm_vec], dim=-1)
        logits = self.classifier(fused)
        return logits
