"""Sequential DistilBERT → BiLSTM → masked pool → classifier (phase6_hybrid_fused §6)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import DistilBertModel


class HybridFusedClassifier(nn.Module):
    """
    DistilBERT token embeddings → BiLSTM → masking-aware mean pool → MLP (3-class risk).

    Forward inputs
    --------------
    input_ids      : LongTensor [B, L] — DistilBERT subword IDs (padded)
    attention_mask : LongTensor [B, L] — 1 for real tokens, 0 for pad

    Tensor flow (defaults: L=256, bert_dim=768, lstm_hidden=64)
    --------------
    last_hidden_state : [B, L, bert_dim]  — DistilBERT encoder output
    packed BiLSTM I/O — variable-length (pack ignores pad positions)
    lstm_seq (padded) : [B, L, 2 * lstm_hidden]  — BiLSTM sequence output
    pooled            : [B, 2 * lstm_hidden]     — masked mean over LSTM outputs
    logits            : [B, num_labels]           — default num_labels=3 (legit / warning / fraud)
    """

    def __init__(
        self,
        lstm_hidden: int = 64,
        fusion_hidden: int = 256,
        num_labels: int = 3,
        dropout: float = 0.3,
        distilbert_name: str = "distilbert-base-uncased",
    ):
        super().__init__()

        self.bert = DistilBertModel.from_pretrained(distilbert_name)
        bert_dim = int(
            getattr(self.bert.config, "hidden_size", None) or self.bert.config.dim
        )

        self.lstm = nn.LSTM(
            bert_dim,
            lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        lstm_out_dim = lstm_hidden * 2

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, fusion_hidden),
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
    ) -> torch.Tensor:
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = bert_out.last_hidden_state

        lengths = attention_mask.long().sum(dim=1).clamp(min=1)
        packed = pack_padded_sequence(
            seq,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        lstm_seq, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=seq.size(1),
        )
        pooled = self._mean_pool(lstm_seq, attention_mask)
        return self.classifier(pooled)
