import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
from PIL import Image, ImageStat, UnidentifiedImageError
import numpy as np
import mlflow
import hashlib
import zipfile


def hash_image(image_path):
    """
    Compute an MD5 hash for an image to detect duplicates.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: MD5 hash string, or None if the file cannot be read.
    """
    try:
        with open(image_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"Failed to hash image {image_path}: {e}")
        return None


def plot_brightness_outliers(df, img_dir, output_path, title):
    """
    Visualize brightness outlier images in a grid.

    Args:
        df (pd.DataFrame): DataFrame containing 'file' and 'brightness' columns.
        img_dir (str): Directory containing images.
        output_path (str): Path to save the grid image.
        title (str): Title for the plot.
    """
    if df.empty:
        print("No brightness outliers to display.")
        return
    cols = min(5, len(df))
    rows = (len(df) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()
    for ax, (_, row) in zip(axes, df.iterrows()):
        img_path = os.path.join(img_dir, row['file'])
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                ax.imshow(img, cmap='gray')
                ax.set_title(f"{row['file']}\nBrightness:{row['brightness']:.1f}", fontsize=7)
                ax.axis('off')
            except UnidentifiedImageError:
                ax.set_title(f"Unreadable: {row['file']}", fontsize=6)
                ax.axis('off')
        else:
            ax.set_title(f"Missing: {row['file']}", fontsize=6)
            ax.axis('off')
    for ax in axes[len(df):]:
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    mlflow.log_artifact(output_path)


def explore_dataset(csv_path, img_dir, sample_size=1000, output_dir='outputs', experiment_name="DatasetExploration"):
    """
    Perform dataset exploration:
    - Class distribution
    - Image size & aspect ratio analysis
    - Brightness analysis (with outlier detection)
    - Duplicate, corrupted, and missing image detection
    - Sample grid visualization
    - Save all outputs and log artifacts to MLflow

    Args:
        csv_path (str): Path to CSV file with 'file' and 'glasses' columns.
        img_dir (str): Directory containing images.
        sample_size (int): Number of images to sample for analysis.
        output_dir (str): Directory to save analysis outputs.
        experiment_name (str): MLflow experiment name.

    Returns:
        tuple: (list of generated files, path to zipped artifacts)
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    if not mlflow.active_run():
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()

    df = pd.read_csv(csv_path)
    if 'file' not in df.columns or 'glasses' not in df.columns:
        raise ValueError("CSV must contain 'file' and 'glasses' columns.")

    print(f"Total images: {len(df)}")
    saved_files = []

    # ===== Class Distribution =====
    class_counts = df['glasses'].value_counts()
    plt.figure(figsize=(6, 4))
    class_counts.plot(kind='bar', title='Class Distribution', color=['skyblue', 'orange'])
    plt.xlabel('Class (0 = No Glasses, 1 = Glasses)')
    plt.ylabel('Count')
    path = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(path)
    plt.close()
    saved_files.append(path)
    mlflow.log_artifact(path)
    for label, count in class_counts.items():
        mlflow.log_param(f'class_{label}_count', count)

    # ===== Image Sizes =====
    widths, heights = [], []
    for img_name in df['file'].sample(min(sample_size, len(df)), random_state=42):
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path)
            widths.append(img.width)
            heights.append(img.height)
        except UnidentifiedImageError:
            print(f"Unreadable image skipped: {img_path}")
            continue

    if widths and heights:
        plt.figure(figsize=(8, 4))
        plt.hist(widths, bins=30, alpha=0.6, label='Widths')
        plt.hist(heights, bins=30, alpha=0.6, label='Heights')
        plt.legend()
        plt.title('Image Dimension Distribution')
        plt.xlabel('Pixels')
        plt.ylabel('Frequency')
        path = os.path.join(output_dir, 'image_dimensions.png')
        plt.savefig(path)
        plt.close()
        saved_files.append(path)
        mlflow.log_artifact(path)

        mlflow.log_param('width_min', min(widths))
        mlflow.log_param('width_max', max(widths))
        mlflow.log_param('height_min', min(heights))
        mlflow.log_param('height_max', max(heights))

    # ===== Aspect Ratio Analysis =====
    aspect_ratios = [w / h for w, h in zip(widths, heights) if h > 0]
    if aspect_ratios:
        plt.figure(figsize=(6, 4))
        plt.hist(aspect_ratios, bins=30, color='purple', alpha=0.7)
        plt.title("Aspect Ratio Distribution")
        plt.xlabel("Width / Height")
        plt.ylabel("Frequency")
        path = os.path.join(output_dir, 'aspect_ratios.png')
        plt.savefig(path)
        plt.close()
        saved_files.append(path)
        mlflow.log_artifact(path)

    # ===== Brightness Analysis =====
    brightness_data = []
    sampled_rows = df.sample(min(sample_size, len(df)), random_state=42)
    for _, row in sampled_rows.iterrows():
        img_path = os.path.join(img_dir, row['file'])
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert('L')
            stat = ImageStat.Stat(img)
            brightness_data.append((row['file'], row['glasses'], stat.mean[0]))
        except UnidentifiedImageError:
            print(f"Unreadable image skipped in brightness analysis: {img_path}")
            continue

    brightness_df = pd.DataFrame(brightness_data, columns=['file', 'glasses', 'brightness'])
    if not brightness_df.empty:
        plt.figure(figsize=(6, 4))
        brightness_df.boxplot(by='glasses', column='brightness', grid=False)
        plt.title("Brightness Distribution by Class")
        plt.suptitle('')
        plt.xlabel('Glasses (0 = No, 1 = Yes)')
        plt.ylabel('Brightness')
        path = os.path.join(output_dir, 'brightness_boxplot.png')
        plt.savefig(path)
        plt.close()
        saved_files.append(path)
        mlflow.log_artifact(path)

        # Detect brightness outliers
        low_thresh = brightness_df['brightness'].quantile(0.05)
        high_thresh = brightness_df['brightness'].quantile(0.95)
        outlier_df = brightness_df[(brightness_df['brightness'] < low_thresh) | (brightness_df['brightness'] > high_thresh)]
        outlier_csv = os.path.join(output_dir, 'brightness_outliers.csv')
        outlier_df.to_csv(outlier_csv, index=False)
        mlflow.log_artifact(outlier_csv)
        print(f"Saved brightness outlier info to {outlier_csv}")

        # Visualize brightness outliers
        brightness_grid = os.path.join(output_dir, 'brightness_outlier_images_grid.png')
        plot_brightness_outliers(outlier_df, img_dir, brightness_grid, title="Brightness Outlier Images")

    # ===== Duplicate & Corrupted Image Detection =====
    hashes = {}
    duplicates, corrupted, missing = [], [], []
    for img_name in df['file']:
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            missing.append(img_name)
            continue
        try:
            Image.open(img_path).verify()
            img_hash = hash_image(img_path)
            if img_hash in hashes:
                duplicates.append(img_name)
            else:
                hashes[img_hash] = img_name
        except UnidentifiedImageError:
            corrupted.append(img_name)

    mlflow.log_param("missing_images_count", len(missing))
    mlflow.log_param("corrupted_images_count", len(corrupted))
    mlflow.log_param("duplicate_images_count", len(duplicates))

    for lst, name in [(missing, "missing_images.txt"), (corrupted, "corrupted_images.txt"), (duplicates, "duplicate_images.txt")]:
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            f.write("\n".join(lst))
        saved_files.append(path)
        mlflow.log_artifact(path)

    # ===== Sample Grid =====
    sample_images = df.sample(min(10, len(df)), random_state=42)
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for ax, (_, row) in zip(axes.flatten(), sample_images.iterrows()):
        img_path = os.path.join(img_dir, row['file'])
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(f"Label: {row['glasses']}")
                ax.axis('off')
            except UnidentifiedImageError:
                ax.axis('off')
    plt.tight_layout()
    path = os.path.join(output_dir, 'sample_grid.png')
    plt.savefig(path)
    plt.close()
    saved_files.append(path)
    mlflow.log_artifact(path)

    # ===== Zip all artifacts =====
    zip_path = os.path.join(output_dir, "dataset_exploration.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in saved_files:
            zipf.write(file, os.path.basename(file))
    mlflow.log_artifact(zip_path)
    print(f"All exploration artifacts zipped at: {zip_path}")

    print("Exploration complete. Outputs saved to:", output_dir)
    return saved_files, zip_path