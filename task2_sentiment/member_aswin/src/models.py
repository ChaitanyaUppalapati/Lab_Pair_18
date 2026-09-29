"""Neural network architectures for Task 2 Yelp Polarity Sentiment Classification.
Includes:
1. Baseline: Trainable Embedding -> Masked Mean Pooling -> FC -> ReLU -> Dropout -> Output
2. Experimental 1: Trainable Embedding -> BiGRU -> Dual Pooling (Max+Avg) -> FC -> Dropout -> Output
3. Experimental 2: Trainable Embedding -> Multi-Scale 1D TextCNN -> Max-over-Time -> FC -> Dropout -> Output

All embeddings are randomly initialized and learned strictly from scratch.
"""

from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineMeanPooling(nn.Module):
    """Baseline Model: Trainable Embedding -> Mean Pooling -> Dense -> ReLU -> Dropout -> Binary Logit."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        output_dim: int = 1,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # Random normal initialization with small std
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        self.embedding.weight.data[padding_idx].zero_()

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)

        # Mask padding tokens (index 0) so they don't corrupt the mean
        mask = (input_ids != 0).unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        masked_sum = (embedded * mask).sum(dim=1)      # (batch_size, embed_dim)
        mask_count = mask.sum(dim=1).clamp(min=1.0)    # (batch_size, 1)
        pooled = masked_sum / mask_count               # (batch_size, embed_dim)

        hidden = self.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden).squeeze(-1)          # (batch_size,)
        return logits


class BiGRUClassifier(nn.Module):
    """Experimental Model 1: Trainable Embedding -> BiGRU -> Max+Mean Pooling -> Dense -> Dropout -> Binary Logit."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.4,
        output_dim: int = 1,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        self.embedding.weight.data[padding_idx].zero_()

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        gru_out_dim = hidden_dim * 2  # Bidirectional doubles output channels
        self.fc1 = nn.Linear(gru_out_dim * 2, hidden_dim)  # Dual pooling (max + avg)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        gru_out, _ = self.gru(embedded)       # (batch_size, seq_len, 2 * hidden_dim)

        mask = (input_ids != 0).unsqueeze(-1)  # (batch_size, seq_len, 1)

        # Masked Average Pooling
        mask_f = mask.float()
        avg_pool = (gru_out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        # Masked Max Pooling
        masked_gru_out = gru_out.masked_fill(~mask, -1e9)
        max_pool = masked_gru_out.max(dim=1)[0]

        # Concatenate dual pooling representations
        pooled = torch.cat([max_pool, avg_pool], dim=1)  # (batch_size, 4 * hidden_dim)

        hidden = self.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden).squeeze(-1)
        return logits


class TextCNNClassifier(nn.Module):
    """Experimental Model 2: Trainable Embedding -> Multi-Scale 1D Convolutions -> Max-over-Time -> Dense -> Dropout -> Binary Logit."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 100,
        filter_sizes: List[int] = [3, 4, 5],
        dense_dim: int = 128,
        dropout: float = 0.4,
        output_dim: int = 1,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        self.embedding.weight.data[padding_idx].zero_()

        # Parallel 1D Convolutions for n-gram feature detection
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2,
            )
            for k in filter_sizes
        ])

        total_features = num_filters * len(filter_sizes)
        self.fc1 = nn.Linear(total_features, dense_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(dense_dim, output_dim)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids)           # (batch_size, seq_len, embed_dim)
        # Conv1d expects (batch_size, in_channels, seq_len)
        embedded = embedded.transpose(1, 2)            # (batch_size, embed_dim, seq_len)

        conv_features = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))          # (batch_size, num_filters, seq_len)
            # Global 1D Max-over-Time Pooling
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)  # (batch_size, num_filters)
            conv_features.append(pooled)

        concatenated = torch.cat(conv_features, dim=1) # (batch_size, num_filters * len(filter_sizes))
        hidden = self.relu(self.fc1(concatenated))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden).squeeze(-1)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    """Instantiate model specified by the configuration file."""
    arch = config["architecture"]
    model_type = arch["model_type"].lower()

    if model_type in ["baseline", "baseline_mean_pooling"]:
        return BaselineMeanPooling(
            vocab_size=vocab_size,
            embedding_dim=arch.get("embedding_dim", 128),
            hidden_dim=arch.get("hidden_dim", 128),
            dropout=arch.get("dropout", 0.3),
            output_dim=arch.get("output_dim", 1),
        )
    elif model_type in ["bigru", "gru", "experimental_1"]:
        return BiGRUClassifier(
            vocab_size=vocab_size,
            embedding_dim=arch.get("embedding_dim", 128),
            hidden_dim=arch.get("hidden_dim", 128),
            num_layers=arch.get("num_layers", 1),
            dropout=arch.get("dropout", 0.4),
            output_dim=arch.get("output_dim", 1),
        )
    elif model_type in ["text_cnn", "cnn", "experimental_2"]:
        return TextCNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=arch.get("embedding_dim", 128),
            num_filters=arch.get("num_filters", 100),
            filter_sizes=arch.get("filter_sizes", [3, 4, 5]),
            dense_dim=arch.get("dense_dim", 128),
            dropout=arch.get("dropout", 0.4),
            output_dim=arch.get("output_dim", 1),
        )
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Choose from 'baseline_mean_pooling', 'bigru', 'text_cnn'.")
