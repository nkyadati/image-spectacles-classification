import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import os
import pandas as pd
import mlflow
import json
import numpy as np
import random
from model import BACKBONE_IMG_SIZES  # Import mapping from model.py


def set_seed(seed=42):
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Random seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for loading image classification data.

    Expects a CSV file with at least two columns:
        - 'file': image file name
        - 'glasses': label (0 or 1)

    Args:
        csv_file (str): Path to CSV file containing image paths and labels.
        img_dir (str): Directory containing the image files.
        transform (callable, optional): Transformations to apply to each image.
    """

    def __init__(self, csv_file, img_dir, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        self.data = pd.read_csv(csv_file)
        required_columns = {'file', 'glasses'}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(f"CSV file must contain columns: {required_columns}")
        
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """Return total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve an image and label by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image tensor, label)
        """
        img_name = self.data.iloc[idx]['file']
        label = int(self.data.iloc[idx]['glasses'])
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        try:
            image = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            raise ValueError(f"Corrupted or unsupported image file: {img_path}")

        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(csv_file, img_dir, batch_size=32, val_split=0.2, seed=42, backbone="resnet50"):
    """
    Prepare training and validation DataLoaders with reproducible splits.

    Args:
        csv_file (str): Path to CSV file containing 'file' and 'glasses' columns.
        img_dir (str): Directory containing the image files.
        batch_size (int): Number of samples per batch (default: 32).
        val_split (float): Fraction of dataset to use for validation (default: 0.2).
        seed (int): Random seed for reproducibility (default: 42).
        backbone (str): Backbone model name (affects image resize size).

    Returns:
        tuple: (train_loader, val_loader, val_df)
            - train_loader: DataLoader for training.
            - val_loader: DataLoader for validation.
            - val_df: Pandas DataFrame of validation samples (for analysis).
    """
    set_seed(seed)

    # Auto-adjust image size based on backbone
    img_size = BACKBONE_IMG_SIZES.get(backbone.lower(), 224)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])

    # Build dataset
    dataset = CustomDataset(csv_file, img_dir, transform=transform)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check CSV file and image directory.")

    # Split into train/val
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("Invalid split sizes. Adjust val_split or provide more data.")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    # Save split indices for reproducibility
    split_indices = {
        "train_indices": train_dataset.indices,
        "val_indices": val_dataset.indices
    }
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/split_indices.json', 'w') as f:
        json.dump(split_indices, f)
    mlflow.log_artifact('outputs/split_indices.json')

    # Log label distribution
    df = pd.read_csv(csv_file)
    counts = df['glasses'].value_counts().to_dict()
    for label, count in counts.items():
        mlflow.log_param(f'label_count_{label}', count)
    mlflow.log_param('train_size', train_size)
    mlflow.log_param('val_size', val_size)
    mlflow.log_param('backbone_image_size', img_size)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Return validation subset as DataFrame (for error analysis)
    val_df = df.iloc[val_dataset.indices]

    return train_loader, val_loader, val_df