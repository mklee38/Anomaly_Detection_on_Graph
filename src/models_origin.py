# src/models.py

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class ImprovedGraphSAGE(nn.Module):
    """
    Improved GraphSAGE model for binary classification on the Elliptic dataset.
    
    All hyperparameters are passed explicitly (no internal dependency on cfg).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.2,
        aggregator: str = "mean"
    ):
        """
        Args:
            in_channels (int): Number of input features per node.
            hidden_dim (int): Hidden layer dimension.
            num_layers (int): Number of GraphSAGE layers (must be >= 1).
            dropout (float): Dropout rate.
            aggregator (str): Aggregation method ('mean', 'lstm', 'pool', etc.).
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.aggregator = aggregator

        # Build convolution layers
        self.convs = nn.ModuleList()

        # First layer: in_channels → hidden_dim
        self.convs.append(
            SAGEConv(in_channels, hidden_dim, aggr=aggregator)
        )

        # Middle layers (if num_layers > 2)
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_dim, hidden_dim, aggr=aggregator)
            )

        # Last SAGEConv layer (if num_layers > 1)
        if num_layers > 1:
            self.convs.append(
                SAGEConv(hidden_dim, hidden_dim, aggr=aggregator)
            )

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_dim, 2)   # Binary classification: licit vs illicit

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Node features [num_nodes, in_channels]
            edge_index (torch.Tensor): Edge indices [2, num_edges]

        Returns:
            torch.Tensor: Output logits [num_nodes, 2]
        """
        # All layers except the last one: Conv → ReLU → Dropout
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
            x = self.dropout(x)

        # Last conv layer: no ReLU
        if len(self.convs) > 0:
            x = self.convs[-1](x, edge_index)

        x = self.dropout(x)
        x = self.lin(x)          # Final linear layer for classification

        return x

    def __repr__(self):
        return (f"ImprovedGraphSAGE("
                f"in={self.convs[0].in_channels}, "
                f"hidden={self.hidden_dim}, "
                f"layers={self.num_layers}, "
                f"agg={self.aggregator}, "
                f"dropout={self.dropout_rate})")