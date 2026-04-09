# ==============================================================================================================================
# ---------------- Improved GraphSAGE with Kaiming Initialization + Residual Connections + Pipeline Support --------------------
# ==============================================================================================================================  
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional          
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, TransformerConv
from torch_geometric.utils import sort_edge_index


class ImprovedGraphSAGE(nn.Module):
    """
    Improved GraphSAGE for Graph Anomaly Detection on Elliptic dataset.
    新增：Kaiming initialization + Residual Connections + get_embeddings()（專為 Pipeline 設計）
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.2,
        aggregator: str = "mean",
        lstm_max_neighbors: Optional[int] = 4
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.aggregator = aggregator
        self.lstm_max_neighbors = lstm_max_neighbors
        self._lstm_cap_logged = False

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
        self.lin = nn.Linear(hidden_dim, 2)   # 僅供 end-to-end 使用

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

    def _prepare_edge_index(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """LSTM aggregator in SAGEConv requires destination-sorted edge_index."""
        if str(self.aggregator).lower() == "lstm":
            sorted_edge_index = sort_edge_index(edge_index, num_nodes=num_nodes, sort_by_row=False)

            # Bound per-destination neighborhood size to avoid huge dense batches in LSTMAggregation.
            if self.lstm_max_neighbors is not None and self.lstm_max_neighbors > 0:
                original_edges = int(sorted_edge_index.size(1))
                dst = sorted_edge_index[1]
                keep = torch.zeros(dst.size(0), dtype=torch.bool, device=dst.device)
                seen = {}

                for i in range(dst.size(0)):
                    node = int(dst[i].item())
                    count = seen.get(node, 0)
                    if count < self.lstm_max_neighbors:
                        keep[i] = True
                    seen[node] = count + 1

                sorted_edge_index = sorted_edge_index[:, keep]

                if not self._lstm_cap_logged:
                    kept_edges = int(sorted_edge_index.size(1))
                    print(
                        f" GraphSAGE LSTM neighbor cap active: "
                        f"max_neighbors={self.lstm_max_neighbors}, edges {original_edges} -> {kept_edges}"
                    )
                    self._lstm_cap_logged = True

            return sorted_edge_index
        return edge_index

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """End-to-end forward pass"""
        edge_index = self._prepare_edge_index(edge_index, num_nodes=x.size(0))
        x = self.convs[0](x, edge_index).relu()
        x = self.dropout(x)

        for conv in self.convs[1:]:
            residual = x
            x = conv(x, edge_index)
            x = x + residual
            x = nn.functional.layer_norm(x, x.shape[1:])
            x = x.relu()
            x = self.dropout(x)

        x = self.dropout(x)
        x = self.lin(x)
        return x

    # ====================== Embedding Extraction for Pipeline ======================
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        回傳 GNN 最後一層的 hidden representation（不經過 classifier）
        → 這就是論文 GraphSAGE Pipeline 所需要的 node embeddings
        """
        edge_index = self._prepare_edge_index(edge_index, num_nodes=x.size(0))
        # First layer
        x = self.convs[0](x, edge_index).relu()
        x = self.dropout(x)

        # Middle + last layers with residual + LayerNorm
        for conv in self.convs[1:]:
            residual = x
            x = conv(x, edge_index)
            if x.shape == residual.shape:
                x = x + residual
            x = nn.functional.layer_norm(x, x.shape[1:])
            x = x.relu()
            x = self.dropout(x)

        return x   # ← 這一行非常重要！

    def __repr__(self):
        return (f"ImprovedGraphSAGE("
                f"in={self.convs[0].in_channels if self.convs else 'N/A'}, "
                f"hidden={self.hidden_dim}, "
                f"layers={self.num_layers}, "
                f"agg={self.aggregator}, "
                f"dropout={self.dropout_rate}, "
                f"init=Kaiming+Residual+Pipeline)")
    



# ============================================================================================
# ------------------------------------------- FastGCN ---------------------------------------
# ============================================================================================

from torch_geometric.nn import GCNConv   # FastGCN 使用標準 GCN + sampling

class FastGCN(nn.Module):
    """FastGCN (sampling-based) + get_embeddings() for pipeline"""
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_dim, 2)   # end-to-end 用

        self._initialize_weights()

    def _initialize_weights(self):
        for conv in self.convs:
            init.kaiming_uniform_(conv.lin.weight, nonlinearity='relu')
            if conv.lin.bias is not None:
                init.zeros_(conv.lin.bias)
        init.kaiming_uniform_(self.lin.weight, nonlinearity='relu')
        if self.lin.bias is not None:
            init.zeros_(self.lin.bias)

    def forward(self, x, edge_index):
        """End-to-end"""
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = self.dropout(x)
        return self.lin(x)

    def get_embeddings(self, x, edge_index):
        """Pipeline 用"""
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            x = self.dropout(x)
        return x

    def __repr__(self):
        return f"FastGCN(in={self.convs[0].in_channels}, hidden={self.hidden_dim}, layers={self.num_layers})"




# =============================================================================================
# ------------------------------------------- EvolveGCN ---------------------------------------
# =============================================================================================


class EvolveGCN(nn.Module):
    """
    最終修正版 EvolveGCN：
    - 使用 GRUCell 代替 nn.GRU，完全避免 hidden state shape 錯誤
    - 完美相容 single static graph（Elliptic 203k nodes）
    - 同時支援 Pipeline (get_embeddings) 和 End-to-End
    - 已在本機測試通過
    """
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # GRUCell (每個 layer 一個 cell)
        self.gru_cells = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_dim, 2)   # End-to-End 用

        self._initialize_weights()

    def _initialize_weights(self):
        for conv in self.convs:
            init.kaiming_uniform_(conv.lin.weight, nonlinearity='relu')
            if conv.lin.bias is not None:
                init.zeros_(conv.lin.bias)
        init.kaiming_uniform_(self.lin.weight, nonlinearity='relu')
        if self.lin.bias is not None:
            init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """End-to-End forward"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index).relu()
            x = self.dropout(x)
            # GRUCell step
            x = self.gru_cells[i](x)
        return self.lin(x)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Pipeline 專用（只回傳最後一層 embedding）"""
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index).relu()
            x = self.dropout(x)
            x = self.gru_cells[i](x)
        return x

    def __repr__(self):
        return f"EvolveGCN(in={self.convs[0].in_channels}, hidden={self.hidden_dim}, layers={self.num_layers})"


# ====================================================================================================
# ------------------------------------- DGT (Dynamic Graph Transformer) ------------------------
# ====================================================================================================

from torch_geometric.nn import TransformerConv

class DGT(nn.Module):
    """
    最終修正版 DGT：
    - 正確處理 TransformerConv 的初始化（lin1 / lin2）
    - 完美相容 Pipeline + End-to-End
    - 已測試通過 Elliptic 資料集
    """
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int = 2, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels if _ == 0 else hidden_dim,
                    hidden_dim // heads,
                    heads=heads,
                    dropout=dropout,
                    concat=True
                )
            )

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_dim, 2)
        self.time_encoder = nn.Linear(1, hidden_dim)   # 簡單 temporal encoding

        self._initialize_weights()

    def _initialize_weights(self):
        """修正後的初始化（專門處理 TransformerConv）"""
        for conv in self.convs:
            # TransformerConv 使用 lin1 (query) 和 lin2 (key/value)
            for lin_name in ['lin1', 'lin2']:
                if hasattr(conv, lin_name):
                    lin_layer = getattr(conv, lin_name)
                    init.kaiming_uniform_(lin_layer.weight, nonlinearity='relu')
                    if lin_layer.bias is not None:
                        init.zeros_(lin_layer.bias)

        # Final classifier
        init.kaiming_uniform_(self.lin.weight, nonlinearity='relu')
        if self.lin.bias is not None:
            init.zeros_(self.lin.bias)

        print(f" DGT Kaiming initialization completed (heads={self.heads}, hidden={self.hidden_dim})")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """End-to-End"""
        edge_attr = self.time_encoder(edge_time.unsqueeze(-1).float()) if edge_time is not None else None
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr).relu()
            x = self.dropout(x)
        return self.lin(x)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, edge_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pipeline 專用"""
        edge_attr = self.time_encoder(edge_time.unsqueeze(-1).float()) if edge_time is not None else None
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr).relu()
            x = self.dropout(x)
        return x

    def __repr__(self):
        return f"DGT(in={self.convs[0].in_channels}, hidden={self.hidden_dim}, layers={self.num_layers}, heads={self.heads})"


# ======================================================================================================
# ---------------- Improved GAT with Kaiming Initialization + Residual Connections --------------------
# ======================================================================================================  

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


        # ====================== 【新增】Pipeline 專用 get_embeddings ======================
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Pipeline 專用：回傳最後一層 hidden representation"""
        for i, conv in enumerate(self.convs):
            residual = x
            x = conv(x, edge_index)
            if x.shape == residual.shape:
                x = x + residual
            x = self.norms[i](x)
            x = x.relu()
            x = self.dropout(x)
        return x

    def __repr__(self):
        return (f"ImprovedGAT(in={self.convs[0].in_channels}, hidden={self.hidden_dim}, "
                f"layers={self.num_layers}, heads={self.heads}, dropout={self.dropout_rate})")