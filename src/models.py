# ====================== Improved GraphSAGE with Kaiming Initialization + Residual Connections ======================   
import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import SAGEConv

class ImprovedGraphSAGE(nn.Module):
    """
    Improved GraphSAGE with Residual Connections for Graph Anomaly Detection on Elliptic dataset.
    新增：Kaiming initialization + Residual (Skip) Connections
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.2,
        aggregator: str = "mean"
    ):
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

        # Middle + Last layers
        for _ in range(num_layers - 1):
            self.convs.append(
                SAGEConv(hidden_dim, hidden_dim, aggr=aggregator)
            )

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_dim, 2)   # Binary classification

        # ========== Kaiming Initialization ==========
        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming (He) initialization for ReLU"""
        for conv in self.convs:
            if hasattr(conv.lin_l, 'weight'):
                init.kaiming_uniform_(conv.lin_l.weight, mode='fan_in', nonlinearity='relu')
            if hasattr(conv.lin_r, 'weight'):
                init.kaiming_uniform_(conv.lin_r.weight, mode='fan_in', nonlinearity='relu')
            
            if hasattr(conv.lin_l, 'bias') and conv.lin_l.bias is not None:
                init.zeros_(conv.lin_l.bias)
            if hasattr(conv.lin_r, 'bias') and conv.lin_r.bias is not None:
                init.zeros_(conv.lin_r.bias)

        init.kaiming_uniform_(self.lin.weight, mode='fan_in', nonlinearity='relu')
        if self.lin.bias is not None:
            init.zeros_(self.lin.bias)

        print(f" Kaiming initialization + Residual Connections applied "
              f"(layers={self.num_layers}, hidden={self.hidden_dim})")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Residual Connections + Layer Normalization
        """
        # First layer (no residual, dimension may differ)
        x = self.convs[0](x, edge_index).relu()
        x = self.dropout(x)

        # Subsequent layers with residual + LayerNorm
        for conv in self.convs[1:]:
            residual = x
            
            x = conv(x, edge_index)
            x = x + residual                    # Residual
            x = nn.functional.layer_norm(x, x.shape[1:])  # LayerNorm (重要！)
            x = x.relu()
            x = self.dropout(x)

        x = self.dropout(x)
        x = self.lin(x)
        return x

    def __repr__(self):
        return (f"ImprovedGraphSAGE("
                f"in={self.convs[0].in_channels if self.convs else 'N/A'}, "
                f"hidden={self.hidden_dim}, "
                f"layers={self.num_layers}, "
                f"agg={self.aggregator}, "
                f"dropout={self.dropout_rate}, "
                f"init=Kaiming+Residual)")
    

# ====================== Improved GAT with Kaiming Initialization + Residual Connections ======================
import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.nn import GATConv


class ImprovedGAT(nn.Module):
    """
    Improved GAT for Elliptic Graph Anomaly Detection.
    已修正：GATConv 初始化 + multi-head 維度處理
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.4
    ):
        super().__init__()

        self.model_name = "GAT"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(
            GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout, concat=True)
        )
        self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # Last layer (不 concat)
        self.convs.append(
            GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=False)
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_dim, 2)

        self._initialize_weights()

    def _initialize_weights(self):
        """安全的 Kaiming initialization for GATConv"""
        for conv in self.convs:
            # GATConv 權重可能在 lin_src / lin_dst / lin
            for attr_name in ['lin_src', 'lin_dst', 'lin']:
                if hasattr(conv, attr_name):
                    lin_layer = getattr(conv, attr_name)
                    if hasattr(lin_layer, 'weight') and lin_layer.weight is not None:
                        init.kaiming_uniform_(lin_layer.weight, nonlinearity='relu')
                    if hasattr(lin_layer, 'bias') and lin_layer.bias is not None:
                        init.zeros_(lin_layer.bias)

        # Final classifier
        init.kaiming_uniform_(self.lin.weight, nonlinearity='relu')
        if self.lin.bias is not None:
            init.zeros_(self.lin.bias)

        print(f" Kaiming initialization applied to ImprovedGAT "
              f"(layers={self.num_layers}, heads={self.heads}, hidden={self.hidden_dim})")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            residual = x
            
            x = conv(x, edge_index)
            
            if x.shape == residual.shape:
                x = x + residual
            
            x = self.norms[i](x)
            x = x.relu()
            x = self.dropout(x)

        x = self.lin(x)
        return x

    def __repr__(self):
        return (f"ImprovedGAT(in={self.convs[0].in_channels}, hidden={self.hidden_dim}, "
                f"layers={self.num_layers}, heads={self.heads}, dropout={self.dropout_rate})")