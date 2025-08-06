import torch
import pandas as pd
import os
import mlflow
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np


def plot_image_grid(df, img_dir, output_path, title, num_samples=None):
    """
    Plot a grid of sample images with labels, predictions, and confidence scores.

    Args:
        df (pd.DataFrame): DataFrame containing 'file', 'true_label', 'predicted_label', 'confidence'.
        img_dir (str): Directory containing the images.
        output_path (str): Path to save the plotted grid.
        title (str): Title for the plot.
        num_samples (int, optional): Number of images to sample. Defaults to showing all.

    Raises:
        FileNotFoundError: If image directory does not exist.
    """
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if df.empty:
        print("No data to plot for:", title)
        return

    if num_samples:
        df = df.sample(min(num_samples, len(df)), random_state=42)
    cols = min(5, len(df))
    rows = (len(df) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, df.iterrows()):
        img_path = os.path.join(img_dir, row['file'])
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(
                    f"ID:{row['file']}\nT:{row['true_label']} P:{row['predicted_label']}\nConf:{row['confidence']:.2f}",
                    fontsize=8
                )
                ax.axis('off')
            except UnidentifiedImageError:
                ax.set_title(f"Unreadable: {row['file']}", fontsize=8)
                ax.axis('off')
        else:
            ax.set_title(f"Missing: {row['file']}", fontsize=8)
            ax.axis('off')
    for ax in axes[len(df):]:
        ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    mlflow.log_artifact(output_path)


def plot_confusion_matrix(y_true, y_pred, output_path, class_names):
    """
    Plot and save a confusion matrix.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        output_path (str): Path to save the confusion matrix image.
        class_names (list): List of class names for labeling the matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path, dpi=300)
    plt.close()
    mlflow.log_artifact(output_path)


def plot_confidence_histogram(correct_confidences, incorrect_confidences, output_path):
    """
    Plot and save confidence distributions for correct vs. incorrect predictions.

    Args:
        correct_confidences (list): Confidence scores for correctly predicted samples.
        incorrect_confidences (list): Confidence scores for incorrectly predicted samples.
        output_path (str): Path to save the histogram image.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(correct_confidences, bins=20, alpha=0.6, label='Correct', color='green')
    plt.hist(incorrect_confidences, bins=20, alpha=0.6, label='Incorrect', color='red')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
    plt.legend()
    plt.savefig(output_path, dpi=300)
    plt.close()
    mlflow.log_artifact(output_path)


def run_error_analysis(model, val_csv, val_img_dir, output_dir='outputs', num_samples=20, device='cpu'):
    """
    Run comprehensive error analysis on validation data.

    Args:
        model (torch.nn.Module): Trained model (with confidence outputs).
        val_csv (str): CSV containing validation data ('file' and 'glasses' columns).
        val_img_dir (str): Directory containing validation images.
        output_dir (str): Directory to save analysis results.
        num_samples (int): Number of images to visualize in sample grids.
        device (str): Device to run inference ('cpu' or 'cuda').

    Returns:
        tuple: Paths to CSVs containing misclassified and suspected noisy samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isfile(val_csv):
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")
    if not os.path.isdir(val_img_dir):
        raise FileNotFoundError(f"Image directory not found: {val_img_dir}")

    if not mlflow.active_run():
        mlflow.set_experiment("ErrorAnalysis")
        mlflow.start_run()

    # Load validation data
    val_df = pd.read_csv(val_csv)
    required_columns = {'file', 'glasses'}
    if not required_columns.issubset(val_df.columns):
        raise ValueError(f"Validation CSV must contain columns: {required_columns}")

    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    errors, correct, noisy_candidates = [], [], []
    correct_confidences, incorrect_confidences = [], []
    all_preds, all_labels = [], []

    with torch.no_grad():
        for _, row in val_df.iterrows():
            img_path = os.path.join(val_img_dir, row['file'])
            if not os.path.exists(img_path):
                print(f"Skipping missing file: {img_path}")
                continue
            try:
                image = Image.open(img_path).convert('RGB')
            except UnidentifiedImageError:
                print(f"Skipping unreadable file: {img_path}")
                continue

            tensor = transform(image).unsqueeze(0).to(device)
            logits, probs = model(tensor)
            pred = torch.argmax(probs, dim=1).item()
            conf = torch.max(probs).item()
            true_label = int(row['glasses'])

            all_preds.append(pred)
            all_labels.append(true_label)

            if pred != true_label:
                errors.append((row['file'], true_label, pred, conf))
                incorrect_confidences.append(conf)
                if conf > 0.8:
                    noisy_candidates.append((row['file'], true_label, pred, conf))
            else:
                correct.append((row['file'], true_label, pred, conf))
                correct_confidences.append(conf)

    # Save error and noisy sample details
    error_df = pd.DataFrame(errors, columns=['file', 'true_label', 'predicted_label', 'confidence'])
    error_csv = os.path.join(output_dir, "misclassified_samples.csv")
    error_df.to_csv(error_csv, index=False)
    mlflow.log_artifact(error_csv)

    noisy_df = pd.DataFrame(noisy_candidates, columns=['file', 'true_label', 'predicted_label', 'confidence'])
    noisy_csv = os.path.join(output_dir, "suspected_noisy_labels.csv")
    noisy_df.to_csv(noisy_csv, index=False)
    mlflow.log_artifact(noisy_csv)

    print(f"Saved misclassified samples to {error_csv}")
    print(f"Saved suspected noisy samples to {noisy_csv}")

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=['No Glasses', 'Glasses'])
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # Plot confusion matrix and confidence histogram
    plot_confusion_matrix(all_labels, all_preds, os.path.join(output_dir, "confusion_matrix.png"), ['No Glasses', 'Glasses'])
    plot_confidence_histogram(correct_confidences, incorrect_confidences, os.path.join(output_dir, "confidence_histogram.png"))

    # Separate high and low-confidence errors
    high_conf_errors = error_df[error_df['confidence'] > 0.8]
    low_conf_errors = error_df[error_df['confidence'] <= 0.8]
    high_conf_csv = os.path.join(output_dir, "high_confidence_errors.csv")
    low_conf_csv = os.path.join(output_dir, "low_confidence_errors.csv")
    high_conf_errors.to_csv(high_conf_csv, index=False)
    low_conf_errors.to_csv(low_conf_csv, index=False)
    mlflow.log_artifact(high_conf_csv)
    mlflow.log_artifact(low_conf_csv)

    # Visualize errors and noisy samples
    if not error_df.empty:
        plot_image_grid(error_df, val_img_dir, os.path.join(output_dir, "misclassified_samples_grid.png"), title="Misclassified Samples", num_samples=num_samples)
    if not noisy_df.empty:
        plot_image_grid(noisy_df, val_img_dir, os.path.join(output_dir, "suspected_noisy_samples_grid.png"), title="Suspected Noisy Samples", num_samples=None)

    return error_csv, noisy_csv