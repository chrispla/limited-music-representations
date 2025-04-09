import argparse
import importlib
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.dataset import MTAT


def get_model_config(model_type):
    """Import model-specific config and model class."""
    try:
        # Dynamically import the model and config
        config_module = importlib.import_module(f"models.{model_type.lower()}.config")
        model_module = importlib.import_module(f"models.{model_type.lower()}.model")

        # Get the model class based on model type
        if model_type.lower() == "vgg":
            model_class = model_module.ShortChunkCNN
        elif model_type.lower() == "musicnn":
            model_class = model_module.Musicnn
        elif model_type.lower() == "ast":
            model_class = model_module.AST
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return config_module.config, model_class
    except ImportError as e:
        raise ImportError(f"Failed to import {model_type} model or config: {e}")


def create_model(model_type, args):
    """Create model instance based on model type."""
    _, model_class = get_model_config(model_type)

    if model_type.lower() == "vgg":
        return model_class()
    elif model_type.lower() == "musicnn":
        return model_class(
            args=args,
            y_input_dim=args.n_mels,
            timbral_k_height=[0.4, 0.7],
            temporal_k_width=[32, 64, 128],
            filter_factor=1.6,
            pool_type="temporal",
        )
    elif model_type.lower() == "ast":
        return model_class()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_sample_input(model_type, device, args):
    """Create appropriate sample input tensor for model graph visualization."""
    if model_type.lower() == "vgg":
        return torch.randn(1, 1, 59049).to(device)
    elif model_type.lower() == "musicnn":
        return torch.randn(1, 1, args.n_mels, 1280).to(device)
    elif model_type.lower() == "ast":
        return torch.randn(1, 1, 128, 1000).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_loss_function(model_type):
    """Get appropriate loss function based on model type."""
    if model_type.lower() == "ast":
        return nn.BCEWithLogitsLoss()
    else:
        return nn.BCELoss()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Unified model training script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["vgg", "musicnn", "ast"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data/MTAT/",
        help="Path to the dataset directory",
    )

    # First parse just the model type to get the right config
    temp_args, _ = parser.parse_known_args()
    model_config, _ = get_model_config(temp_args.model)

    # Add model-specific config parameters
    for k, v in model_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v) if v is not None else str)

    # Parse all arguments
    args = parser.parse_args()

    # Setup run name and directories
    model_type = args.model.lower()
    run_name = datetime.now().strftime(f"{model_type}_%Y%m%d_%H%M%S")
    if hasattr(args, "tracks_per_genre") and hasattr(args, "epochs"):
        run_name = f"{model_type}_tpg{args.tracks_per_genre}_epoch{args.epochs}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log_dir = Path("./logs") / run_name
    checkpoint_dir = Path("./ckpt") / run_name
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Log config parameters
    for k, v in vars(args).items():
        writer.add_text("config", f"{k}: {v}")

    # Create model
    model = create_model(model_type, args)
    print(f"Created {model_type.upper()} model")
    print("Number of params:", sum(p.numel() for p in model.parameters()))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create dataset and dataloader
    dataset = MTAT(
        root=args.dataset_dir,
        args=args,
        download=False,
    )
    loader = dataset.get_dataloader(batch_size=args.batch_size, shuffle=True)

    # Log dataset information
    writer.add_scalar("Dataset/total_items", len(dataset), 0)
    writer.add_scalar("Dataset/total_tracks", len(dataset.track_ids), 0)

    # Setup loss function, optimizer and scheduler
    loss_function = get_loss_function(model_type)
    optimizer = Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    # Optional: Add model graph to TensorBoard
    try:
        sample_input = get_sample_input(model_type, device, args)
        writer.add_graph(model, sample_input)
    except Exception as e:
        print(f"Failed to add model graph to TensorBoard: {e}")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
        for i, data in pbar:
            inputs, labels = data

            # Handle AST's half-precision labels
            if model_type.lower() == "ast":
                labels = labels.half()

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            current_loss = running_loss / (i + 1)
            pbar.set_description(f"Epoch {epoch + 1}, Loss: {current_loss:.4f}")

            # Log metrics to tensorboard (every 10 batches)
            if i % 10 == 0:
                global_step = epoch * len(loader) + i
                writer.add_scalar("Training/Loss", current_loss, global_step)
                writer.add_scalar(
                    "Training/LearningRate",
                    optimizer.param_groups[0]["lr"],
                    global_step,
                )

        # Log epoch metrics
        writer.add_scalar("Training/EpochLoss", running_loss / len(loader), epoch)

        scheduler.step(running_loss)

        # Save model checkpoint
        save_interval = 10 if model_type.lower() == "ast" else 50
        if (epoch + 1) % save_interval == 0:
            model_path = (
                checkpoint_dir / f"model_{epoch}_loss_{format(running_loss, '.3f')}.pth"
            )
            torch.save(model.state_dict(), model_path)
            writer.add_text(
                "Checkpoints", f"Saved model at epoch {epoch}: {model_path}"
            )

    writer.close()
    print(f"Training completed. Model checkpoints saved to {checkpoint_dir}")


if __name__ == "__main__":
    main()
