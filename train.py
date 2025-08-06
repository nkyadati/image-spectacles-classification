import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import mlflow
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import os


def plot_loss(train_losses, val_losses, output_path='outputs/loss_curve.png'):
    """
    Plot and save training & validation loss curves.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        output_path (str): File path to save the plot.
    """
    try:
        plt.figure()
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss Curve')
        plt.legend()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        mlflow.log_artifact(output_path)
    except Exception as e:
        print(f"Warning: Failed to plot loss curve. {e}")


def log_frozen_layers(model, phase="initial"):
    """
    Log the number and names of frozen layers to MLflow.

    Args:
        model (torch.nn.Module): The model being trained.
    """
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    mlflow.log_param(f"frozen_layers_count_{phase}", len(frozen))
    mlflow.log_text("\n".join(frozen), f"frozen_layers_{phase}.txt")


def train_model(model,
                dataloader,
                val_dataloader=None,
                epochs=20,
                lr=0.001,
                device='cpu',
                experiment_name="ClasssificationExperiment",
                backbone="resnet50",
                head_epochs=5,
                fine_tune_lr=1e-4,
                patience=5):
    """
    Train a model using two-phase fine-tuning:
    - Phase 1: Train only the classifier head for 'head_epochs'.
    - Phase 2: Unfreeze the entire backbone and fine-tune with a smaller learning rate.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): Training data loader.
        val_dataloader (torch.utils.data.DataLoader): Validation data loader (optional).
        epochs (int): Total number of epochs to train.
        lr (float): Learning rate for head-only training.
        device (str): Device to train on ('cpu' or 'cuda').
        experiment_name (str): MLflow experiment name.
        backbone (str): Model backbone name.
        head_epochs (int): Number of epochs for head-only training before unfreezing backbone.
        fine_tune_lr (float): Learning rate for fine-tuning the full network.
        patience (int): Early stopping patience.

    Returns:
        torch.nn.Module: Trained model with best weights loaded.
    """
    if dataloader is None or len(dataloader) == 0:
        raise ValueError("Training dataloader is empty or None.")

    model = model.to(device)
    criterion = CrossEntropyLoss()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    counter = 0
    train_losses, val_losses = [], []

    # Start MLflow run
    if not mlflow.active_run():
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
    mlflow.log_params({
        'epochs': epochs,
        'learning_rate': lr,
        'fine_tune_lr': fine_tune_lr,
        'backbone': backbone,
        'head_epochs': head_epochs,
        'early_stopping_patience': patience
    })

    # === Phase 1: Head-only fine-tuning ===
    for param in model.parameters():
        param.requires_grad = False
    if not hasattr(model, "fc"):
        raise AttributeError("Model does not have an attribute 'fc'. Ensure the architecture defines it.")
    for param in model.fc.parameters():
        param.requires_grad = True
    log_frozen_layers(model, phase="head_only")

    optimizer = optim.Adam(model.fc.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    for epoch in range(epochs):
        # === Phase switch: Unfreeze backbone after head_epochs ===
        if epoch == head_epochs:
            print(f"Unfreezing backbone for full fine-tuning at epoch {epoch + 1}...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=fine_tune_lr)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
            log_frozen_layers(model, phase="full_finetune")

        # === Training ===
        model.train()
        total_loss = 0
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(dataloader)
        train_losses.append(avg_train_loss)

        # === Validation ===
        val_loss = avg_train_loss
        if val_dataloader is not None:
            model.eval()
            val_total_loss = 0
            with torch.no_grad():
                for images, labels in val_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    logits, _ = model(images)
                    val_loss_batch = criterion(logits, labels)
                    val_total_loss += val_loss_batch.item()
            val_loss = val_total_loss / len(val_dataloader)
        val_losses.append(val_loss)

        # Step scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Logging
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.6f}")
        mlflow.log_metrics({
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr
        }, step=epoch)

        # Early stopping and checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, f'outputs/best_{backbone}.pth')
            mlflow.log_artifact(f'outputs/best_{backbone}.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                mlflow.log_param('early_stopping_epoch', epoch + 1)
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), f'outputs/final_{backbone}.pth')
    mlflow.log_artifact(f'outputs/final_{backbone}.pth')
    plot_loss(train_losses, val_losses)
    return model