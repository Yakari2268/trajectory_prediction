import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
from trajectoryDataset import TrajectoryDataset
from trajectoryPredictor import TrajectoryPredictor

# Assume TrajectoryPredictor and TrajectoryDataset classes are defined as above

# --- 1. Hyperparameters and Configuration ---
INPUT_DIM = 3
HIDDEN_DIM = 128
OUTPUT_DIM = 3
NUM_LAYERS = 3
DROPOUT_PROB = 0.5
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 1000
SCHEDULER_STEP_SIZE = 50
SCHEDULER_GAMMA = 0.1
EARLY_STOPPING_PATIENCE = 100

# --- 2. Data Loading ---
train_dataset = TrajectoryDataset('/kaggle/working/train_segments.npz')
val_dataset = TrajectoryDataset('/kaggle/working/val_segments.npz')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Model, Optimizer, and Loss Function Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrajectoryPredictor(
    INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    NUM_LAYERS,
    DROPOUT_PROB
).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
criterion = nn.MSELoss()

# --- 4. Early Stopping and Training Loop ---
best_val_loss = float('inf')
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()
    total_train_loss = 0
    
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        
        future_len = y_batch.size(1)
        predictions = model(x_batch, future_len)
        
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / max(len(train_loader), 1)
    train_losses.append(avg_train_loss)
    
    # --- Validation Phase ---
    model.eval()
    total_val_loss = 0
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            future_len = y_batch.size(1)
            predictions = model(x_batch, future_len)
            loss = criterion(predictions, y_batch)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / max(len(val_loader), 1)
    val_losses.append(avg_val_loss)
    
    print(f"[{epoch+1:03d}/{NUM_EPOCHS}] "
          f"Train Loss: {avg_train_loss:.8f} | "
          f"Val Loss: {avg_val_loss:.8f}")
    
    # --- Scheduler and Early Stopping ---
    scheduler.step()
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"✅ Val loss improved to {best_val_loss:.8f}. Model saved.")
    else:
        patience_counter += 1
        print(f"⚠️ Val loss did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
    
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("⏹️ Early stopping triggered.")
        break

print("Training complete.")

# Optionally save the loss curves for later plotting
np.savez("training_curves.npz", train_losses=train_losses, val_losses=val_losses)
print("Saved loss curves to training_curves.npz.")
