#!/usr/bin/env python3.13
"""
scripts/train_gru_head.py
=========================
Train a PyTorch GRU model for pitch sequence prediction.

This script builds a sequence model that uses recent pitch history to predict
the next pitch type, complementing the tree-based models with sequential patterns.

Architecture:
- Embedding layer for pitch type IDs (vocab size 10, embedding dim 16)
- GRU layer (input 83, hidden 64, 1 layer)
- Fully connected output layer (64 → 9 pitch types)

Features:
- Last 5 pitch type IDs (padded with 9 for missing)
- balls, strikes (count state)
- dvelo1 (velocity change)

Usage:
    python scripts/train_gru_head.py --train-years 2023 --val-range 2024-04-01:2024-04-15
    GPU=1 python scripts/train_gru_head.py --train-years 2023 --val-range 2024-04-01:2024-04-15
"""

import argparse
import sys
import pathlib
import numpy as np
import pandas as pd
import os

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from run_full_pipeline import load_parquets, add_family_probs, add_temporal_weight

# Import PyTorch modules
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import accuracy_score, log_loss
except ImportError as e:
    print(f"❌ Error: Required PyTorch modules not found: {e}")
    print("💡 Please install PyTorch: pip install torch torchvision torchaudio")
    sys.exit(1)

# --- Config ---
MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Check for GPU availability
USE_GPU = os.getenv("GPU", "0") == "1" and torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
print(f"🖥️  Training on {'GPU' if USE_GPU else 'CPU'}")


class PitchSequenceDataset(Dataset):
    """Dataset for pitch sequence prediction."""

    def __init__(self, sequences, context, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.context = torch.tensor(context, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.context[idx], self.targets[idx]


class PitchGRU(nn.Module):
    """GRU model for pitch sequence prediction."""

    def __init__(self, vocab_size=10, embed_dim=16, gru_hidden=64, num_classes=9):
        super(PitchGRU, self).__init__()

        # Embedding for pitch type IDs
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # GRU layer (embed_dim * 5 + 3 context features = 83 input features)
        self.gru = nn.GRU(
            input_size=embed_dim * 5 + 3,  # 5 embedded pitches + 3 context
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(gru_hidden, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, sequences, context):
        # sequences: (batch_size, 5) - last 5 pitch IDs
        # context: (batch_size, 3) - balls, strikes, dvelo1

        batch_size = sequences.size(0)

        # Embed pitch sequences
        embedded = self.embedding(sequences)  # (batch_size, 5, embed_dim)
        embedded = embedded.view(batch_size, -1)  # (batch_size, 5 * embed_dim)

        # Concatenate with context
        features = torch.cat(
            [embedded, context], dim=1
        )  # (batch_size, 5*embed_dim + 3)

        # Add sequence dimension for GRU
        features = features.unsqueeze(1)  # (batch_size, 1, features)

        # Pass through GRU
        gru_out, _ = self.gru(features)  # (batch_size, 1, hidden)
        gru_out = gru_out.squeeze(1)  # (batch_size, hidden)

        # Apply dropout and output layer
        out = self.dropout(gru_out)
        logits = self.fc(out)  # (batch_size, num_classes)

        return logits


def prepare_sequence_features(df):
    """Prepare sequence features for GRU training."""
    print("🔧 Preparing sequence features...")

    # Create pitch type ID mapping (9 main classes + padding)
    pitch_types = ["FF", "SI", "SL", "CH", "CU", "FC", "FS", "KC", "ST"]
    pitch_to_id = {pt: i for i, pt in enumerate(pitch_types)}
    pitch_to_id["OTHER"] = len(pitch_types)  # OTHER maps to padding ID
    padding_id = 9  # Use 9 as padding for missing pitches

    # Prepare sequences (last 5 pitch types)
    sequence_cols = ["prev_pitch_1", "prev_pitch_2", "prev_pitch_3", "prev_pitch_4"]
    sequences = []

    for _, row in df.iterrows():
        seq = []
        # Add current target as first in sequence for context
        current_pitch = row.get("pitch_type_can", "OTHER")
        seq.append(pitch_to_id.get(current_pitch, padding_id))

        # Add previous pitches
        for col in sequence_cols:
            prev_pitch = row.get(col, None)
            if pd.isna(prev_pitch) or prev_pitch is None:
                seq.append(padding_id)
            else:
                seq.append(pitch_to_id.get(prev_pitch, padding_id))

        sequences.append(seq)

    sequences = np.array(sequences)

    # Prepare context features
    context = df[["balls", "strikes", "dvelo1"]].fillna(0).values.astype(np.float32)

    # Prepare targets (exclude OTHER for 9-class prediction)
    targets = []
    for _, row in df.iterrows():
        pitch = row.get("pitch_type_can", "OTHER")
        if pitch in pitch_to_id and pitch != "OTHER":
            targets.append(pitch_to_id[pitch])
        else:
            targets.append(-1)  # Mark invalid targets

    targets = np.array(targets)

    # Filter out invalid targets
    valid_mask = targets != -1
    sequences = sequences[valid_mask]
    context = context[valid_mask]
    targets = targets[valid_mask]

    print(f"✅ Prepared {len(sequences)} valid sequences")
    print(f"📊 Sequence shape: {sequences.shape}")
    print(f"📊 Context shape: {context.shape}")
    print(f"📊 Target classes: {len(np.unique(targets))}")

    return sequences, context, targets


def train_gru_model(train_years, val_range, epochs=3):
    """Train the GRU model on specified data."""

    print(f"🚀 Training GRU model")
    print(f"📊 Train years: {train_years}")
    print(f"📈 Validation range: {val_range}")
    print(f"🔄 Epochs: {epochs}")
    print("=" * 60)

    # Load data
    print("⏳ Loading training data...")
    train_df = load_parquets(train_years)

    print("⏳ Loading validation data...")
    val_year = int(val_range.split(":")[0].split("-")[0])
    val_df = load_parquets([val_year], val_range)

    print("⏳ Loading test data...")
    # Use a small test range after validation
    test_start = val_range.split(":")[1]  # End of validation
    year, month, day = test_start.split("-")
    next_day = int(day) + 1
    test_range = f"{year}-{month}-{next_day:02d}:{year}-{month}-30"
    test_df = load_parquets([val_year], test_range)

    # Add family probabilities and temporal weighting
    print("⚙️  Adding family probabilities and temporal weights...")
    latest_date = pd.to_datetime(train_df["game_date"].max())
    train_df = add_temporal_weight(train_df, latest_date, 0.0008)
    train_df = add_family_probs(train_df)
    val_df = add_family_probs(val_df)
    test_df = add_family_probs(test_df)

    # Prepare features
    train_seq, train_ctx, train_tgt = prepare_sequence_features(train_df)
    val_seq, val_ctx, val_tgt = prepare_sequence_features(val_df)
    test_seq, test_ctx, test_tgt = prepare_sequence_features(test_df)

    # Create datasets and dataloaders
    train_dataset = PitchSequenceDataset(train_seq, train_ctx, train_tgt)
    val_dataset = PitchSequenceDataset(val_seq, val_ctx, val_tgt)
    test_dataset = PitchSequenceDataset(test_seq, test_ctx, test_tgt)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    # Initialize model
    model = PitchGRU(vocab_size=10, embed_dim=16, gru_hidden=64, num_classes=9)
    model = model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"🎯 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    print("🚀 Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (sequences, context, targets) in enumerate(train_loader):
            sequences = sequences.to(DEVICE)
            context = context.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(sequences, context)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                )

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for sequences, context, targets in val_loader:
                sequences = sequences.to(DEVICE)
                context = context.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(sequences, context)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

                # Collect probabilities for log-loss
                probs = torch.softmax(outputs, dim=1)
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_logloss = log_loss(val_targets, val_probs)

        print(f"📊 Epoch {epoch+1}/{epochs}:")
        print(f"   Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(
            f"   Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val LogLoss: {val_logloss:.4f}"
        )

    # Generate logits for validation and test sets
    print("📊 Generating validation logits...")
    model.eval()
    val_logits = []
    with torch.no_grad():
        for sequences, context, targets in val_loader:
            sequences = sequences.to(DEVICE)
            context = context.to(DEVICE)
            outputs = model(sequences, context)
            val_logits.extend(outputs.cpu().numpy())

    print("📊 Generating test logits...")
    test_logits = []
    with torch.no_grad():
        for sequences, context, targets in test_loader:
            sequences = sequences.to(DEVICE)
            context = context.to(DEVICE)
            outputs = model(sequences, context)
            test_logits.extend(outputs.cpu().numpy())

    # Save model and logits
    model_path = MODEL_DIR / "gru_head.pt"
    val_logits_path = MODEL_DIR / "gru_logits_val.npy"
    test_logits_path = MODEL_DIR / "gru_logits_test.npy"

    torch.save(model.state_dict(), model_path)
    np.save(val_logits_path, np.array(val_logits))
    np.save(test_logits_path, np.array(test_logits))

    print(f"💾 Model saved to: {model_path}")
    print(f"💾 Val logits saved to: {val_logits_path}")
    print(f"💾 Test logits saved to: {test_logits_path}")
    print("✅ GRU training complete!")

    return model, val_logits, test_logits


def main():
    parser = argparse.ArgumentParser(
        description="Train GRU head model for pitch prediction"
    )
    parser.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        required=True,
        help="Years to use for training (e.g., --train-years 2023 2024)",
    )
    parser.add_argument(
        "--val-range",
        type=str,
        required=True,
        help="Validation date range (e.g., 2024-04-01:2024-04-15)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs (default: 3)"
    )

    args = parser.parse_args()

    train_gru_model(
        train_years=args.train_years, val_range=args.val_range, epochs=args.epochs
    )


if __name__ == "__main__":
    main()
