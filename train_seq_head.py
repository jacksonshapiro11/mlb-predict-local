#!/usr/bin/env python
"""
train_seq_head.py
================
PyTorch GRU sequence model for MLB pitch prediction.
Uses last 5 pitches (types + count + velocity changes) to predict next pitch.

USAGE
-----
python train_seq_head.py
"""

import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import duckdb
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore")

# GPU Configuration
USE_GPU = os.getenv("GPU", "0") == "1"
print(f"🖥️  Training on {'GPU' if USE_GPU else 'CPU'}")

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
PARQUET_DIR = pathlib.Path("data/features_historical")
MODEL_DIR = pathlib.Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PT = "pitch_type_can"
SEQ_LEN = 5
BATCH_SIZE = 512
EPOCHS = 3
LR = 1e-3

# Pitch type mapping (8 = OTHER for padding)
PITCH_TYPES = ["FF", "SL", "CH", "CU", "SI", "FC", "FS", "KC", "OTHER"]

LAG_SQL = """
WITH base AS (
  SELECT *
  FROM parquet_scan({paths})
  {where_clause}
)
SELECT *, 
       LAG(pitch_type_can,1) OVER w AS prev_pt1,
       LAG(pitch_type_can,2) OVER w AS prev_pt2,
       LAG(pitch_type_can,3) OVER w AS prev_pt3,
       LAG(pitch_type_can,4) OVER w AS prev_pt4,
       LAG(pitch_type_can,5) OVER w AS prev_pt5,
       LAG(balls,1) OVER w AS prev_balls1,
       LAG(balls,2) OVER w AS prev_balls2,
       LAG(balls,3) OVER w AS prev_balls3,
       LAG(balls,4) OVER w AS prev_balls4,
       LAG(balls,5) OVER w AS prev_balls5,
       LAG(strikes,1) OVER w AS prev_strikes1,
       LAG(strikes,2) OVER w AS prev_strikes2,
       LAG(strikes,3) OVER w AS prev_strikes3,
       LAG(strikes,4) OVER w AS prev_strikes4,
       LAG(strikes,5) OVER w AS prev_strikes5,
       release_speed - LAG(release_speed,1) OVER w AS dvelo1,
       LAG(release_speed,1) OVER w - LAG(release_speed,2) OVER w AS dvelo2,
       LAG(release_speed,2) OVER w - LAG(release_speed,3) OVER w AS dvelo3,
       LAG(release_speed,3) OVER w - LAG(release_speed,4) OVER w AS dvelo4,
       LAG(release_speed,4) OVER w - LAG(release_speed,5) OVER w AS dvelo5
FROM base
WINDOW w AS (
  PARTITION BY pitcher, game_pk
  ORDER BY at_bat_number, pitch_number
)
"""


# --------------------------------------------------------------------------- #
# DATA LOADING
# --------------------------------------------------------------------------- #
def load_duck(query: str) -> pd.DataFrame:
    con = duckdb.connect()
    df = con.execute(query).df()
    con.close()
    return df


def load_parquets(years, date_range: str | None = None):
    paths = [str(PARQUET_DIR / f"statcast_historical_{y}.parquet") for y in years]
    path_expr = "[" + ",".join([f"'{p}'" for p in paths]) + "]"
    where_clause = ""
    if date_range:
        start, end = date_range.split(":")
        where_clause = f"WHERE game_date BETWEEN DATE '{start}' AND DATE '{end}'"
    q = LAG_SQL.format(paths=path_expr, where_clause=where_clause)
    print(f"🗄️  Loading data: {years} {date_range or 'all dates'}")
    return load_duck(q)


# --------------------------------------------------------------------------- #
# DATASET
# --------------------------------------------------------------------------- #
class PitchSequenceDataset(Dataset):
    def __init__(self, df, target_encoder):
        self.df = df.copy()
        self.target_encoder = target_encoder

        # Create sequences
        self.sequences = []
        self.targets = []

        print("🔄 Building sequences...")
        for idx in tqdm(range(len(df))):
            row = df.iloc[idx]

            # Get sequence of last 5 pitch types (pad with 8='OTHER')
            pitch_seq = []
            for i in range(1, SEQ_LEN + 1):
                pt = row.get(f"prev_pt{i}")
                if pd.isna(pt):
                    pitch_seq.append(8)  # OTHER = 8 for padding
                else:
                    try:
                        pitch_seq.append(PITCH_TYPES.index(pt))
                    except ValueError:
                        pitch_seq.append(8)  # Unknown pitch type -> OTHER

            # Get sequence of last 5 balls (pad with 0)
            balls_seq = []
            for i in range(1, SEQ_LEN + 1):
                balls = row.get(f"prev_balls{i}")
                balls_seq.append(0 if pd.isna(balls) else int(balls))

            # Get sequence of last 5 strikes (pad with 0)
            strikes_seq = []
            for i in range(1, SEQ_LEN + 1):
                strikes = row.get(f"prev_strikes{i}")
                strikes_seq.append(0 if pd.isna(strikes) else int(strikes))

            # Get sequence of last 5 velocity changes (pad with 0)
            dvelo_seq = []
            for i in range(1, SEQ_LEN + 1):
                dvelo = row.get(f"dvelo{i}")
                dvelo_seq.append(0.0 if pd.isna(dvelo) else float(dvelo))

            # Reverse sequences (oldest to newest)
            pitch_seq.reverse()
            balls_seq.reverse()
            strikes_seq.reverse()
            dvelo_seq.reverse()

            # Target
            target = row[TARGET_PT]
            if pd.isna(target) or target not in PITCH_TYPES:
                continue

            target_idx = PITCH_TYPES.index(target)

            self.sequences.append(
                {
                    "pitch_types": pitch_seq,
                    "balls": balls_seq,
                    "strikes": strikes_seq,
                    "dvelo": dvelo_seq,
                }
            )
            self.targets.append(target_idx)

        print(f"✅ Created {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]

        # Convert to tensors
        pitch_types = torch.LongTensor(seq["pitch_types"])
        balls = torch.FloatTensor(seq["balls"])
        strikes = torch.FloatTensor(seq["strikes"])
        dvelo = torch.FloatTensor(seq["dvelo"])
        target = torch.LongTensor([target])

        return {
            "pitch_types": pitch_types,
            "balls": balls,
            "strikes": strikes,
            "dvelo": dvelo,
            "target": target,
        }


# --------------------------------------------------------------------------- #
# MODEL
# --------------------------------------------------------------------------- #
class PitchGRU(nn.Module):
    def __init__(self, vocab_size=9, embed_dim=6, hidden_size=64, num_classes=9):
        super(PitchGRU, self).__init__()

        # Embedding for pitch types
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # GRU layer (input: embed_dim + 3 for balls/strikes/dvelo)
        self.gru = nn.GRU(
            input_size=embed_dim + 3,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, pitch_types, balls, strikes, dvelo):
        # Embed pitch types
        embedded = self.embedding(pitch_types)  # (batch, seq, embed_dim)

        # Stack additional features
        balls = balls.unsqueeze(-1)  # (batch, seq, 1)
        strikes = strikes.unsqueeze(-1)  # (batch, seq, 1)
        dvelo = dvelo.unsqueeze(-1)  # (batch, seq, 1)

        # Concatenate all features
        x = torch.cat(
            [embedded, balls, strikes, dvelo], dim=-1
        )  # (batch, seq, embed_dim+3)

        # GRU forward
        gru_out, hidden = self.gru(x)  # gru_out: (batch, seq, hidden_size)

        # Use last timestep output
        last_output = gru_out[:, -1, :]  # (batch, hidden_size)

        # Final prediction
        logits = self.fc(last_output)  # (batch, num_classes)

        return logits


# --------------------------------------------------------------------------- #
# TRAINING
# --------------------------------------------------------------------------- #
def train_model():
    # Load data
    print("📊 Loading training data (2019-2023)...")
    train_years = [2019, 2020, 2021, 2022, 2023]
    train_df = load_parquets(train_years)

    print("📊 Loading validation data (2024 Apr-Jul)...")
    val_df = load_parquets([2024], "2024-04-01:2024-07-31")

    print("📊 Loading test data (2024 Aug-Dec)...")
    test_df = load_parquets([2024], "2024-08-01:2024-12-31")

    # Create target encoder
    target_encoder = LabelEncoder()
    target_encoder.fit(PITCH_TYPES)

    # Create datasets
    train_dataset = PitchSequenceDataset(train_df, target_encoder)
    val_dataset = PitchSequenceDataset(val_df, target_encoder)
    test_dataset = PitchSequenceDataset(test_df, target_encoder)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(
        f"📊 Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # Initialize model
    device = torch.device("cuda" if USE_GPU else "cpu")
    print(f"🖥️  Using device: {device}")

    model = PitchGRU().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    print(f"\n🚂 Training for {EPOCHS} epochs...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            # Move to device
            pitch_types = batch["pitch_types"].to(device)
            balls = batch["balls"].to(device)
            strikes = batch["strikes"].to(device)
            dvelo = batch["dvelo"].to(device)
            targets = batch["target"].squeeze().to(device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(pitch_types, balls, strikes, dvelo)
            loss = criterion(logits, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100*correct/total:.2f}%"}
            )

        print(
            f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Accuracy={100*correct/total:.2f}%"
        )

    # Save model
    model_path = MODEL_DIR / "gru_sequence_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

    # Generate predictions
    print("\n🔮 Generating predictions...")
    model.eval()

    def get_logits(data_loader, name):
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Predicting {name}"):
                pitch_types = batch["pitch_types"].to(device)
                balls = batch["balls"].to(device)
                strikes = batch["strikes"].to(device)
                dvelo = batch["dvelo"].to(device)
                targets = batch["target"].squeeze().to(device)

                logits = model(pitch_types, balls, strikes, dvelo)

                all_logits.append(logits.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        logits_array = np.vstack(all_logits)
        targets_array = np.concatenate(all_targets)

        # Calculate accuracy
        predictions = np.argmax(logits_array, axis=1)
        accuracy = np.mean(predictions == targets_array)
        print(f"{name} accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")

        return logits_array

    # Save predictions
    val_logits = get_logits(val_loader, "Validation")
    test_logits = get_logits(test_loader, "Test")

    val_path = MODEL_DIR / "gru_val_logits.npy"
    test_path = MODEL_DIR / "gru_test_logits.npy"

    np.save(val_path, val_logits)
    np.save(test_path, test_logits)

    print(f"✅ Validation logits saved to {val_path}")
    print(f"✅ Test logits saved to {test_path}")
    print("🚀 Training complete!")


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    train_model()
