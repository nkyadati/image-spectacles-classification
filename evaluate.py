import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mlflow
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_confusion_matrix(cm, class_names, output_path='outputs/confusion_matrix.png'):
    """
    Plot and save a confusion matrix.

    Args:
        cm (ndarray): Confusion matrix array.
        class_names (list): List of class names for labeling.
        output_path (str): Path to save the confusion matrix plot.
    """
    if cm.size == 0:
        print("Warning: Empty confusion matrix. Skipping plot.")
        return
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    mlflow.log_artifact(output_path)


def plot_confidence_histogram(confidences, output_path='outputs/confidence_histogram.png'):
    """
    Plot and save a histogram of prediction confidences.

    Args:
        confidences (list): List of confidence scores for predictions.
        output_path (str): Path to save the histogram.
    """
    if not confidences:
        print("Warning: No confidences provided. Skipping histogram plot.")
        return
    plt.figure(figsize=(6, 4))
    plt.hist(confidences, bins=20, color='green', alpha=0.7)
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    mlflow.log_artifact(output_path)


def evaluate_model(model, dataloader, device='cpu', class_names=None):
    """
    Evaluate a trained model on a validation/test dataset.

    Args:
        model (torch.nn.Module): Trained model with forward() returning (logits, probabilities).
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (str): Device to run inference ('cpu' or 'cuda').
        class_names (list, optional): List of class names for metrics and plots.

    Returns:
        tuple: (accuracy, confusion_matrix, confidences)
            - accuracy (float): Overall accuracy score.
            - confusion_matrix (ndarray): Confusion matrix array.
            - confidences (list): List of prediction confidences for each sample.
    """
    if not hasattr(model, 'eval'):
        raise TypeError("Model must be a valid PyTorch model.")
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty. Cannot perform evaluation.")

    model.eval()
    all_preds, all_labels, all_confidences = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            try:
                logits, probs = model(images)
            except Exception as e:
                raise RuntimeError(f"Model forward() did not return expected outputs: {e}")

            preds = torch.argmax(probs, dim=1)
            confidences = probs.max(dim=1).values
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())

    if not all_preds or not all_labels:
        raise ValueError("No predictions or labels collected during evaluation.")

    # Generate classification report
    report = classification_report(all_labels, all_preds, output_dict=True)
    os.makedirs('outputs', exist_ok=True)
    report_path = 'outputs/classification_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f)
    mlflow.log_artifact(report_path)
    mlflow.log_metrics({f"class_{k}_f1": v['f1-score'] for k, v in report.items() if isinstance(v, dict)})

    # Accuracy
    acc = accuracy_score(all_labels, all_preds)
    mlflow.log_metric("val_accuracy", acc)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    if class_names is None:
        class_names = [str(i) for i in range(len(set(all_labels)))]
    plot_confusion_matrix(cm, class_names)

    # Confidence histogram
    plot_confidence_histogram(all_confidences)

    print("Evaluation complete.\n", classification_report(all_labels, all_preds))
    return acc, cm, all_confidences