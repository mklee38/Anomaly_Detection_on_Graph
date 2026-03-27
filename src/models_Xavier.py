import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import SAGEConv


class ImprovedGraphSAGE(nn.Module):
    """
    Improved GraphSAGE model for binary classification on the Elliptic dataset.
    新增：Xavier (Glorot) initialization + residual connection 準備
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
            num_layers (int): Number of GraphSAGE layers.
            dropout (float): Dropout rate.
            aggregator (str): Aggregation method ('mean', 'lstm', 'pool').
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

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_dim, hidden_dim, aggr=aggregator)
            )

        # Last layer
        if num_layers > 1:
            self.convs.append(
                SAGEConv(hidden_dim, hidden_dim, aggr=aggregator)
            )

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_dim, 2)   # Binary classification: licit (0) vs illicit (1)

        # ====================== Xavier (Glorot) Initialization ======================
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Xavier uniform initialization to all linear layers in SAGEConv and final classifier"""
        for conv in self.convs:
            # SAGEConv 內部有兩個 linear: lin_l (self) 和 lin_r (neighbor)
            if hasattr(conv.lin_l, 'weight'):
                init.xavier_uniform_(conv.lin_l.weight)
            if hasattr(conv.lin_r, 'weight'):
                init.xavier_uniform_(conv.lin_r.weight)
            
            # 如果有 bias，也初始化為 0
            if hasattr(conv.lin_l, 'bias') and conv.lin_l.bias is not None:
                init.zeros_(conv.lin_l.bias)
            if hasattr(conv.lin_r, 'bias') and conv.lin_r.bias is not None:
                init.zeros_(conv.lin_r.bias)

        # Final linear layer
        init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            init.zeros_(self.lin.bias)

        print(f" Xavier (Glorot) initialization applied to ImprovedGraphSAGE "
              f"(layers={self.num_layers}, hidden={self.hidden_dim})")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional residual connections (目前先保留原本結構，之後可輕鬆加上)
        """
        # All layers except the last one: Conv → ReLU → Dropout
        for conv in self.convs[:-1]:
            x = conv(x, edge_index).relu()
            x = self.dropout(x)

        # Last conv layer
        if len(self.convs) > 0:
            x = self.convs[-1](x, edge_index)

        x = self.dropout(x)
        x = self.lin(x)          # Final classification

        return x

    def __repr__(self):
        return (f"ImprovedGraphSAGE("
                f"in={self.convs[0].in_channels}, "
                f"hidden={self.hidden_dim}, "
                f"layers={self.num_layers}, "
                f"agg={self.aggregator}, "
                f"dropout={self.dropout_rate}, "
                f"init=Xavier)")