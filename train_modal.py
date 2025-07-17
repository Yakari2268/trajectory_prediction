import modal
import os
import pathlib
import sys

# --- 1. Modal App and Environment Setup ---

APP_NAME = "trajectory-prediction-training"
VOLUME_NAME = "trajectory-shared-volume"
VOL_MOUNT_PATH = pathlib.Path("/data")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.3.1", "numpy==1.26.4")
    .add_local_file("trajectoryDataset.py", remote_path="/root/trajectoryDataset.py")
    .add_local_file("trajectoryPredictor.py", remote_path="/root/trajectoryPredictor.py")
)

app = modal.App(APP_NAME, image=image)


# --- 2. Remote Training Function ---

@app.function(
    gpu="H100",
    volumes={VOL_MOUNT_PATH: volume},
    timeout=7200,
)
def train_model():
    """
    This function reads data from and writes checkpoints to the shared volume.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR
    from torch.utils.data import DataLoader
    import numpy as np

    sys.path.append("/root")
    from trajectoryDataset import TrajectoryDataset
    from trajectoryPredictor import TrajectoryPredictor

    # --- Hyperparameters ---
    INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM = 3, 128, 3
    NUM_LAYERS, DROPOUT_PROB = 3, 0.5
    LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS = 0.001, 64, 1000
    SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA = 50, 0.1
    EARLY_STOPPING_PATIENCE = 100

    # --- Paths within the Volume ---
    train_data_path = VOL_MOUNT_PATH / "train_segments.npz"
    val_data_path = VOL_MOUNT_PATH / "val_segments.npz"
    model_save_path = VOL_MOUNT_PATH / "best_model.pth"
    curves_save_path = VOL_MOUNT_PATH / "training_curves.npz"

    # --- Data Loading ---
    # This reload is CORRECT and NECESSARY to see the uploaded files.
    volume.reload()
    print(f"🛰️  Loading datasets from volume path: {VOL_MOUNT_PATH}")
    train_dataset = TrajectoryDataset(train_data_path)
    val_dataset = TrajectoryDataset(val_data_path)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Model, Optimizer, and Loss ---
    device = torch.device("cuda")
    print(f"🏋️  Using device: {device}")
    model = TrajectoryPredictor(
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT_PROB
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    criterion = nn.MSELoss()

    # --- Training Loop ---
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch, y_batch.size(1))
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                predictions = model(x_batch, y_batch.size(1))
                total_val_loss += criterion(predictions, y_batch).item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"[{epoch+1:03d}/{NUM_EPOCHS}] "
            f"Train Loss: {avg_train_loss:.8f} | "
            f"Val Loss: {avg_val_loss:.8f}"
        )

        scheduler.step()
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Val loss improved. Model saved to volume.")
        else:
            patience_counter += 1
            print(f"⚠️ Val loss did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("⏹️ Early stopping triggered.")
            break

    print("\nTraining complete.")
    np.savez(curves_save_path, train_losses=train_losses, val_losses=val_losses)
    print(f"📈 Saved final loss curves to volume.")


# --- 3. Local Entrypoint ---
@app.local_entrypoint()
def main():
    """
    This function runs from your local machine.
    It first uploads data to the volume, then starts the remote training job.
    """
    local_train_path = "./train_segments.npz"
    remote_train_path = "/train_segments.npz"
    local_val_path = "./val_segments.npz"
    remote_val_path = "/val_segments.npz"

    # THE FIX: Remove the check and the reload(). Just upload the data.
    # The batch_upload is idempotent, meaning it overwrites the files,
    # ensuring the latest versions are always available for training.
    print("📦 Uploading datasets to volume...")
    with volume.batch_upload() as batch:
        batch.put_file(local_train_path, remote_train_path)
        batch.put_file(local_val_path, remote_val_path)
    print("✅ Datasets uploaded successfully.")

    # Start the training job on a remote GPU.
    print("\n🚀 Starting remote training job on Modal...")
    train_model.remote()