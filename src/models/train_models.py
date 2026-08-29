"""
Training Pipeline for AI-ZUPT Detector and Motion Context Classifier

Generates annotated multi-sensor datasets and trains:
1. AIZUPTNet: Binary stance classifier (Stationary vs Moving)
2. MotionResNet1D: 5-class motion regime classifier (Stationary, Pedestrian, Vehicular, Aerial, Rover)
"""

import os
import argparse
from typing import Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.simulation.trajectory_gen import TrajectoryGenerator
from src.models.zupt_detector import AIZUPTNet
from src.models.motion_classifier import MotionResNet1D, MotionContext


def generate_zupt_training_data(num_sequences: int = 40,
                                duration_per_seq: float = 30.0,
                                window_size: int = 21,
                                dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize annotated sliding window dataset for AI-ZUPT training.
    """
    print(f"[AI-ZUPT Data] Generating {num_sequences} annotated IMU sequences...")
    all_windows = []
    all_labels = []

    profiles = [
        ("pedestrian", 0.5),      # High proportion of stance phases
        ("stationary", 0.2),      # 100% stance
        ("urban_driving", 0.2),   # Stop-and-go
        ("planetary_rover", 0.1), # Low speed / crawler
    ]

    seq_idx = 0
    for profile_name, ratio in profiles:
        count = max(2, int(num_sequences * ratio))
        for _ in range(count):
            gen = TrajectoryGenerator(
                profile=profile_name,
                duration=duration_per_seq,
                dt=dt,
                imu_grade="consumer" if seq_idx % 2 == 0 else "industrial",
                seed=1000 + seq_idx
            )
            df = gen.generate()
            acc = df[['acc_x', 'acc_y', 'acc_z']].values
            gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values
            stance_mask = df['stance_mask'].values.astype(np.float32)

            feat = np.hstack([acc, gyro])  # (N, 6)
            N = len(df)
            half_w = window_size // 2

            # Extract sliding windows (subsample stride for balance)
            step = 3 if profile_name != "stationary" else 15
            for k in range(half_w, N - half_w, step):
                win = feat[k - half_w : k + half_w + 1].T  # (6, W)
                label = stance_mask[k]
                all_windows.append(win)
                all_labels.append(label)

            seq_idx += 1

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)
    print(f"[AI-ZUPT Data] Extracted {len(X)} windows (Stance ratio: {np.mean(y):.2%})")
    return X, y


def generate_motion_classifier_data(num_sequences: int = 50,
                                    duration_per_seq: float = 30.0,
                                    window_size: int = 100,
                                    dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize annotated sliding window dataset for 5-class Motion Context Classification.
    """
    print(f"[Motion Classifier Data] Generating {num_sequences} sequences across 5 motion regimes...")
    all_windows = []
    all_labels = []

    profile_map = [
        ("stationary", MotionContext.STATIONARY),
        ("pedestrian", MotionContext.PEDESTRIAN),
        ("urban_driving", MotionContext.VEHICULAR),
        ("helical_3d", MotionContext.AERIAL),
        ("planetary_rover", MotionContext.PLANETARY_ROVER),
    ]

    seq_idx = 0
    count_per_class = max(2, num_sequences // len(profile_map))
    for profile_name, context_enum in profile_map:
        for _ in range(count_per_class):
            gen = TrajectoryGenerator(
                profile=profile_name,
                duration=duration_per_seq,
                dt=dt,
                imu_grade="consumer" if seq_idx % 2 == 0 else "industrial",
                seed=2000 + seq_idx
            )
            df = gen.generate()
            acc = df[['acc_x', 'acc_y', 'acc_z']].values
            gyro = df[['gyro_x', 'gyro_y', 'gyro_z']].values

            feat = np.hstack([acc, gyro])  # (N, 6)
            N = len(df)

            # Slice windows of length W
            step = 25
            for k in range(0, N - window_size, step):
                win = feat[k : k + window_size].T  # (6, W)
                all_windows.append(win)
                all_labels.append(context_enum.value)

            seq_idx += 1

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    print(f"[Motion Classifier Data] Extracted {len(X)} windows across {len(profile_map)} classes")
    return X, y


def train_ai_zupt(save_path: str = "src/models/weights/aizupt_net.pth",
                  epochs: int = 15,
                  batch_size: int = 64,
                  lr: float = 1e-3):
    """Train AIZUPTNet binary stance detector."""
    if not TORCH_AVAILABLE:
        print("[NAVIS] PyTorch not installed. AIZUPTNet training skipped (using statistical GLRT/ARE/MAG detectors).")
        return

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================\n[TRAINING] AI-ZUPT Model on {device}\n========================================")

    X, y = generate_zupt_training_data(num_sequences=35, duration_per_seq=25.0, window_size=21)

    # Train/Val split
    indices = np.random.permutation(len(X))
    split = int(0.85 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AIZUPTNet(in_channels=6, window_size=21).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(bx)
            pred_binary = (pred >= 0.5).float()
            correct += (pred_binary == by).sum().item()
            total += len(bx)

        scheduler.step()
        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item() * len(bx)
                pred_binary = (pred >= 0.5).float()
                val_correct += (pred_binary == by).sum().item()
                val_total += len(bx)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"[AI-ZUPT] Saved best model (Val Acc: {best_val_acc:.2%}) to: {save_path}")


def train_motion_classifier(save_path: str = "src/models/weights/motion_resnet1d.pth",
                            epochs: int = 15,
                            batch_size: int = 64,
                            lr: float = 1e-3):
    """Train MotionResNet1D 5-class motion regime classifier."""
    if not TORCH_AVAILABLE:
        print("[NAVIS] PyTorch not installed. MotionResNet1D training skipped (using rule-based kinematics classifier).")
        return

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================\n[TRAINING] Motion Classifier on {device}\n========================================")

    X, y = generate_motion_classifier_data(num_sequences=45, duration_per_seq=25.0, window_size=100)

    # Train/Val split
    indices = np.random.permutation(len(X))
    split = int(0.85 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx]))
    val_ds = TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MotionResNet1D(in_channels=6, num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(bx)
            pred = torch.argmax(logits, dim=-1)
            correct += (pred == by).sum().item()
            total += len(bx)

        scheduler.step()
        train_loss /= total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(bx)
                pred = torch.argmax(logits, dim=-1)
                val_correct += (pred == by).sum().item()
                val_total += len(bx)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"[Motion Classifier] Saved best model (Val Acc: {best_val_acc:.2%}) to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NAVIS AI Models")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    train_ai_zupt(epochs=args.epochs, batch_size=args.batch_size)
    train_motion_classifier(epochs=args.epochs, batch_size=args.batch_size)
