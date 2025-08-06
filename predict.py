import torch
import pandas as pd
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import os


def generate_predictions(model,
                         test_csv="test.csv",
                         img_dir="data/test",
                         output_csv="outputs/submission.csv",
                         device="cpu"):
    """
    Generate predictions (with confidence) using a trained model object and save them to a CSV.

    Args:
        model (torch.nn.Module): Trained model with forward() returning (logits, probabilities).
        test_csv (str): Path to CSV file containing image file names and IDs.
        img_dir (str): Directory containing test images.
        output_csv (str): Path to save predictions in CSV format.
        device (str): Device to run inference ('cpu' or 'cuda').

    Returns:
        str: Path to the saved CSV file with predictions.

    Raises:
        FileNotFoundError: If test_csv does not exist.
        ValueError: If the test CSV is empty or does not contain required columns.
    """
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    df = pd.read_csv(test_csv)
    if df.empty:
        raise ValueError("Test CSV is empty.")
    if not {'id', 'file'}.issubset(df.columns):
        raise ValueError("Test CSV must contain 'id' and 'file' columns.")

    model = model.to(device)
    model.eval()

    # Define preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    predictions = []
    with torch.no_grad():
        for _, row in df.iterrows():
            img_id = row['id']
            img_name = row['file']
            img_path = os.path.join(img_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
            except (FileNotFoundError, UnidentifiedImageError):
                print(f"Warning: Failed to load {img_path}. Using placeholder image.")
                image = Image.new('RGB', (224, 224), color=(0, 0, 0))  # Placeholder for missing images
            tensor = transform(image).unsqueeze(0).to(device)

            try:
                logits, probs = model(tensor)
            except Exception as e:
                raise RuntimeError(f"Model inference failed for {img_name}: {e}")

            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
            predictions.append((img_id, pred, confidence))

    # Save predictions
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    submission = pd.DataFrame(predictions, columns=["Id", "Category", "Confidence"])
    submission.to_csv(output_csv, index=False)
    print(f"Predictions with confidence saved to {output_csv}")
    return output_csv