# src/split.py

import torch
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


def split_data(
    data,
    y: torch.Tensor,
    device: torch.device,
    temporal_split: bool = True,
    test_time_threshold: int = 34,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the Elliptic dataset into train, validation, and test indices.
    
    Supports temporal split (recommended) or random split as fallback.
    
    Args:
        data: PyG Data object (should contain 'time_steps' attribute for temporal split)
        y (torch.Tensor): Labels tensor
        device (torch.device): Device to store the index tensors
        temporal_split (bool): Whether to use temporal split (time <= 34 for train/val)
        test_time_threshold (int): Time step threshold for test set (default: 34)
        val_size (float): Validation set size as fraction of train+val
        random_state (int): Random seed for reproducibility
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (train_idx, val_idx, test_idx)
    """
    
    time_step = getattr(data, 'time_steps', None)

    if temporal_split and time_step is not None:
        # Temporal Split (Recommended for Elliptic dataset)
        time_step = time_step.to(device)
        
        train_val_mask = (time_step <= test_time_threshold) & (y != -1)
        test_mask      = (time_step > test_time_threshold) & (y != -1)

        labeled_indices = torch.where(train_val_mask)[0].cpu().numpy()

        # Split train_val into train and validation with stratification
        train_idx_np, val_idx_np = train_test_split(
            labeled_indices,
            test_size=val_size,
            stratify=y[train_val_mask].cpu().numpy(),
            random_state=random_state
        )

        train_idx = torch.tensor(train_idx_np, device=device)
        val_idx   = torch.tensor(val_idx_np, device=device)
        test_idx  = torch.where(test_mask)[0]

        print("使用 Temporal Split（推薦）")

    else:
        # Random Split (Fallback)
        labeled_mask = y != -1
        labeled_indices = torch.where(labeled_mask)[0].cpu().numpy()

        # First split: train_val vs test
        train_val_idx, test_idx_np = train_test_split(
            labeled_indices,
            test_size=0.2,
            stratify=y[labeled_mask].cpu().numpy(),
            random_state=random_state
        )

        # Second split: train vs validation
        train_idx_np, val_idx_np = train_test_split(
            train_val_idx,
            test_size=val_size,
            stratify=y[train_val_idx].cpu().numpy(),
            random_state=random_state
        )

        train_idx = torch.tensor(train_idx_np, device=device)
        val_idx   = torch.tensor(val_idx_np, device=device)
        test_idx  = torch.tensor(test_idx_np, device=device)

        print("使用 Random Split（fallback）")

    print(f"訓練: {len(train_idx)} | 驗證: {len(val_idx)} | 測試: {len(test_idx)}")

    return train_idx, val_idx, test_idx


def split_data_temporal(
    data,
    y: torch.Tensor,
    device: torch.device,
    test_time_threshold: int = 34,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience function that forces temporal split.
    """
    return split_data(
        data=data,
        y=y,
        device=device,
        temporal_split=True,
        test_time_threshold=test_time_threshold,
        val_size=val_size,
        random_state=random_state
    )