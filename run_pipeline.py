import argparse
import os
import torch
import mlflow
from model import get_model
from dataset import get_dataloaders
from train import train_model
from evaluate import evaluate_model
from predict import generate_predictions
from explore import explore_dataset
from error_analysis import run_error_analysis

def main(args):
    os.makedirs('outputs', exist_ok=True)
    mlflow.set_experiment("ClassificationPipeline")

    with mlflow.start_run(run_name="full_pipeline") as pipeline_run:
        mlflow.log_param("backbone", args.backbone)  # Log backbone at pipeline level

        # === Step 1: Dataset Exploration ===
        print("Exploring dataset...")
        with mlflow.start_run(run_name="dataset_exploration", nested=True):
            saved_files, zip_path = explore_dataset(
                args.train_csv,
                args.image_dir,
                sample_size=1000,
                output_dir='outputs'
            )
            print("Exploration artifacts:", saved_files)
            print(f"Zipped exploration report: {zip_path}")

        # === Step 2: Load Data ===
        print("Loading data...")
        train_loader, val_loader, val_df = get_dataloaders(
            csv_file=args.train_csv,
            img_dir=args.image_dir,
            batch_size=args.batch_size,
            val_split=args.val_split,
            backbone=args.backbone  # Pass to auto-adjust image size
        )
        val_csv_path = 'outputs/validation_split.csv'
        val_df.to_csv(val_csv_path, index=False)
        mlflow.log_artifact(val_csv_path)

        # === Step 3: Initialize Model ===
        print(f"Initializing model with backbone: {args.backbone}")
        model = get_model(backbone=args.backbone, num_classes=2)

        # === Step 4: Train ===
        print("Training model...")
        with mlflow.start_run(run_name="training", nested=True):
            train_model(model, train_loader, val_loader,
                        epochs=args.epochs, lr=args.learning_rate,
                        device=args.device, backbone=args.backbone)

        # === Step 5: Evaluate ===
        print("Evaluating model...")
        with mlflow.start_run(run_name="evaluation", nested=True):
            acc, cm, confidences = evaluate_model(model, val_loader, device=args.device, class_names=['No Glasses', 'Glasses'])
            mlflow.log_metric("final_val_accuracy", acc)
            torch.save(torch.tensor(confidences), "outputs/val_confidences.pt")
            mlflow.log_artifact("outputs/val_confidences.pt")

        # === Step 6: Error Analysis ===
        print("Running error analysis...")
        with mlflow.start_run(run_name="error_analysis", nested=True):
            run_error_analysis(
                model=model,  
                val_csv=val_csv_path,
                val_img_dir=args.image_dir,
                output_dir='outputs',
                num_samples=20,
                device=args.device
            )

        # === Step 7: Predict ===
        print("Generating predictions for submission...")
        with mlflow.start_run(run_name="prediction", nested=True):
            generate_predictions(
                model=model, 
                test_csv=args.test_csv,
                img_dir=args.test_dir,
                output_csv='outputs/submission.csv',
                device=args.device
            )
        mlflow.log_artifact('outputs/submission.csv')

        print("Pipeline completed. Results saved in outputs/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full training, evaluation, and prediction pipeline")
    parser.add_argument('--train_csv', type=str, default='/Users/karthikyadati/image-spectacles-classification/train.csv', help='Path to train.csv')
    parser.add_argument('--image_dir', type=str, default='/Users/karthikyadati/image-spectacles-classification/dataset', help='Directory with training images')
    parser.add_argument('--test_csv', type=str, default='/Users/karthikyadati/image-spectacles-classification/test.csv', help='Path to test.csv')
    parser.add_argument('--test_dir', type=str, default='/Users/karthikyadati/image-spectacles-classification/dataset', help='Directory with test images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split fraction')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--backbone', type=str, default='mobilenetv3',
                        choices=['resnet50', 'mobilenetv3', 'efficientnet_b4'],
                        help='Backbone model architecture')
    args = parser.parse_args()
    main(args)