# src/data.py
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data       # InMemoryDataset 是 PyG 提供的基類，適合一次性讀取整個圖的資料集。Data 是 PyG 的核心資料結構，包含節點特徵、邊列表、標籤等。
from torch_geometric.utils import to_undirected, degree              
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

import os
import numpy as np
from sklearn.preprocessing import StandardScaler



class EllipticDataset(InMemoryDataset):
    """
    適配你的 features.csv (col 0 = txId, col 1 = time step)
    強制跳過 edgelist header，並加強 debug
    """

    def __init__(self,
                 root: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_degree: bool = False,       
                 use_pagerank: bool = False):
        self.use_degree = use_degree                     
        self.use_pagerank = use_pagerank                 
        super().__init__(root, transform, pre_transform, pre_filter)            # 檢查 processed_file 是否存在，如果不存在就呼叫 process()。
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            'elliptic_txs_classes.csv',
            'elliptic_txs_edgelist.csv',
            'elliptic_txs_features.csv'
        ]

    @property
    def processed_file_names(self):
        return 'elliptic_processed.pt'

    def download(self):
        pass

    def process(self):
        print("開始處理 Elliptic 資料集...")
        print("MARKER: process() method called - About to read classes")

        # 讀檔
        classes = pd.read_csv(os.path.join(self.raw_dir, 'elliptic_txs_classes.csv'))
        features = pd.read_csv(os.path.join(self.raw_dir, 'elliptic_txs_features.csv'), header=None)  # features 沒有 header 行，所以 header=None，這樣 pandas 就不會把第一行當成欄位名稱，而是直接當成資料讀進來。

        # 你的格式：col 0 = txId, col 1 = time step, col 2~ = features
        tx_ids = features.iloc[:, 0].values.astype(np.int64)                        # 取第一列作為 txId，轉成 numpy array，並確保是 int64 類型。    
        time_steps = features.iloc[:, 1].values.astype(np.int64)                    # 取第二列作為 time step，轉成 numpy array，並確保是 int64 類型。
        node_features = features.iloc[:, 2:].values.astype(np.float32)              # 從第三列開始是特徵，轉成 numpy array，並確保是 float32 類型，這樣後面轉成 tensor 時就不會占太多記憶體。

        print(f"Nodes: {len(features)}, Features dim: {node_features.shape[1]}")
        print(f"Sample txIds (col 0): {tx_ids[:5].tolist()}")
        print(f"Sample time steps (col 1): {time_steps[:5].tolist()}")


        # 標準化
        scaler = StandardScaler()                                                   # 建立一個 StandardScaler 物件，這是 sklearn 提供的工具，用來對特徵進行標準化處理，使其具有零均值和單位方差。
        node_features = scaler.fit_transform(node_features)                         # Z-score 標準化（均值0、標準差1），這對 GNN 訓練通常有很大幫助。對 node_features 進行標準化，fit_transform() 會先計算每個特徵的均值和標準差，然後對每個特徵進行 (x - mean) / std 的轉換。


        # 標籤：需要按 features 的 txId 順序對齐
        label_map = {'1': 1, '2': 0, 'unknown': -1}
        # 先建立 txId -> label 的映射,  將 elliptic_txs_classes.csv 和 features.csv 的 txId 進行對齐，確保每個 txId 都有對應的 label。
        txid_to_label = {}
        for idx, row in classes.iterrows():   # 逐行讀取 classes 資料框，對每一行的 txId 和 class 進行處理
            tx_id = int(row.iloc[0])  # 第一列是 txId
            class_str = str(row.iloc[1])  # 第二列是 class
            txid_to_label[tx_id] = label_map.get(class_str, -1) # 根據 class_str 從 label_map 中獲取對應的數字標籤，如果 class_str 不在 label_map 中，就預設為 -1（unknown）。
        

        # 按 features 的 txId 順序創建 labels 陣列
        labels = np.array([txid_to_label.get(int(tx), -1) for tx in tx_ids], dtype=np.int64)

        # txId → node index 映射 (key 轉 int)
        tx_to_node_idx = {int(tx): i for i, tx in enumerate(tx_ids)}

        # 讀 edgelist，跳過 header 行
        edges_path = os.path.join(self.raw_dir, 'elliptic_txs_edgelist.csv')
        edges = pd.read_csv(edges_path, skiprows=1, names=['txId1', 'txId2'])

        print(f"edgelist 有效行數: {len(edges)}")
        print(f"Sample edges txId1 (first 5): {edges['txId1'].head().tolist()}"),


        # 建 edge_index + 計數匹配成功率
        src, dst = [], []
        matched = 0
        total = len(edges)
        print("MARKER: Starting edge matching loop")
        for tx1, tx2 in zip(edges['txId1'], edges['txId2']):
            try:
                i1 = tx_to_node_idx.get(int(tx1), -1)
                i2 = tx_to_node_idx.get(int(tx2), -1)
                if i1 >= 0 and i2 >= 0:
                    src.append(i1)
                    dst.append(i2)
                    matched += 1
            except ValueError:
                continue  # 跳過非數字行

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        print(f"有效 directed edges: {edge_index.shape[1]}")
        print(f"匹配成功率: {matched / total * 100:.2f}% ({matched}/{total})")

        if edge_index.shape[1] == 0:
            print("警告：沒有匹配到邊！檢查 txId 是否全為數字且在 features col 0 中。")
            print("Sample txIds from features:", list(tx_to_node_idx.keys())[:10])
            print("Sample txId from edges:", edges['txId1'].unique()[:10])

        # 轉無向
        edge_index = to_undirected(edge_index)
        print(f"無向 edges: {edge_index.shape[1]}")

        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        # === Degree feature ===
        if self.use_degree:
            if edge_index.shape[1] > 0:
                deg = degree(edge_index[0], num_nodes=len(x)).float().unsqueeze(1)
                x = torch.cat([x, deg], dim=1)
                print(f"加入 degree → x dim: {x.shape[1]}")
            else:
                print("無邊，degree 無法計算")

        # === PageRank feature (networkx version - fixed) ===
        if self.use_pagerank:
            if edge_index.shape[1] > 0:
                #import networkx as nx
                #from torch_geometric.utils import to_networkx
                #from torch_geometric.data import Data

                print("正在計算 PageRank (networkx), 請耐心等待...")

                temp_data = Data(edge_index=edge_index, num_nodes=x.size(0))
                G = to_networkx(temp_data, to_undirected=True)

                pr_dict = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-06)

                pr = torch.tensor([pr_dict.get(i, 0.0) for i in range(x.size(0))],
                                  dtype=torch.float32).unsqueeze(1)

                x = torch.cat([x, pr], dim=1)
                print(f"加入 PageRank → x dim: {x.shape[1]}")
            else:
                print("無邊，PageRank 無法計算")



        # Save the processed data - create tuple manually (data_dict, slices, type)
        # This avoids edge loss from collate batching
        data = Data(
            x=x, 
            edge_index=edge_index, 
            y=y, 
            time_steps=torch.from_numpy(time_steps)
        )

        # === 正確的儲存方式 ===
        data_list = [data]                    # 因為只有一張大圖
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])

        print("處理完成！")
        print(f"最終: {data.num_nodes} nodes, {data.num_edges} edges")

    def __repr__(self) -> str:
        return f'EllipticDataset({self.num_nodes} nodes, {self.num_edges} edges)'