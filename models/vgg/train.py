import argparse
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import config
from model import ShortChunkCNN
from ..dataset import MTAT

run_name = datetime.now().strftime(f"vgg_%Y%m%d_%H%M%S")
log_dir = Path("./logs") / run_name
writer = SummaryWriter(log_dir)

parser = argparse.ArgumentParser(description="Training.")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v)
args = parser.parse_args()

# Log config parameters
for k, v in vars(args).items():
    writer.add_text("config", f"{k}: {v}")

model = ShortChunkCNN()
print("Number of params:", sum(p.numel() for p in model.parameters()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MTAT(
    root="./data/MTAT/",
    args=args,
    download=False,
)
loader = dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)

# Log dataset information
writer.add_scalar("Dataset/total_items", len(dataset), 0)
writer.add_scalar("Dataset/total_tracks", len(dataset.track_ids), 0)

loss_function = nn.BCELoss()
optimizer = Adam(
    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
)
# Adaptive learning rate
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=args.lr_patience
)
model.to(device)

checkpoint_dir = Path("./ckpt") / f"vgg_{args.tracks_per_genre}_{args.epochs}_MTAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
checkpoint_dir.mkdir(exist_ok=True, parents=True)

# Optional: Add model graph to TensorBoard
try:
    sample_input = torch.randn(1, 1, 59049).to(device)  # Adjust size based on your model's expected input
    writer.add_graph(model, sample_input)
except Exception as e:
    print(f"Failed to add model graph to TensorBoard: {e}")

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, data in pbar:
        inputs, labels = data

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        current_loss = running_loss / (i + 1)
        pbar.set_description(f"Epoch {epoch+1}, Loss: {current_loss:.4f}")

        # Log metrics to tensorboard (every 10 batches to avoid logging too much)
        if i % 10 == 0:
            global_step = epoch * len(loader) + i
            writer.add_scalar("Training/Loss", current_loss, global_step)
            writer.add_scalar("Training/LearningRate", optimizer.param_groups[0]["lr"], global_step)

    # Log epoch metrics
    writer.add_scalar("Training/EpochLoss", running_loss / len(loader), epoch)
    
    scheduler.step(running_loss)

    # save model
    if (epoch + 1) % 50 == 0:
        model_path = checkpoint_dir / f"model_{epoch}_loss_{format(running_loss, '.3f')}.pth"
        torch.save(model.state_dict(), model_path)
        writer.add_text("Checkpoints", f"Saved model at epoch {epoch}: {model_path}")

writer.close()