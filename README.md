
# **Image Spectacles Classification**

This project provides an **end-to-end pipeline** for an image classification task (detecting whether a person is wearing glasses), with **MLflow integration**, **dataset exploration**, **error analysis**, and **FastAPI API for inference**.

It supports **multiple backbones** (`ResNet50`, `MobileNetV3`, `EfficientNet-B4`), **two-phase fine-tuning**, and outputs **confidence scores** for predictions. Please download the dataset from the [competition](https://www.kaggle.com/competitions/applications-of-deep-learning-wustl-fall-2023/data) page and place it in the `dataset` folder in the root of the repository.

---

## **Features**
- **End-to-end pipeline**:
  - Dataset exploration  
  - Data loading with dynamic preprocessing  
  - Model initialization (**ResNet50**, **MobileNetV3**, **EfficientNet-B4**)  
  - Two-phase fine-tuning (head-only + full network)  
  - Early stopping & best checkpoint saving  
  - Model evaluation with confusion matrix, confidence histograms, and metrics  
  - Error analysis: misclassified samples & noisy label detection  
  - Batch prediction with confidence scores  

- **MLflow integration**:
  - Logs hyperparameters, metrics, artifacts (plots, CSVs, model weights)  
  - Logs dataset exploration results & splits  
  - Tracks all pipeline steps in nested runs  

- **API for Inference**:
  - FastAPI endpoint for serving predictions  
  - Logs inference metadata (file names, timestamps, response time)  

---

## **Project Structure**
```
image-spectacles-classification/
│
├── run_pipeline.py       # End-to-end pipeline (exploration → training → evaluation → predictions)
├── dataset.py            # Dataset loader with auto image resizing & MLflow logging
├── explore.py            # Dataset exploration (integrated into pipeline)
├── model.py              # Backbone-based classification model with confidence outputs
├── train.py              # Two-phase fine-tuning with MLflow logging & early stopping
├── evaluate.py           # Model evaluation with confusion matrix & confidence histograms
├── error_analysis.py     # Misclassification & noisy label analysis with visualizations
├── predict.py            # Batch prediction with confidence scores
├── api.py                # FastAPI endpoint for inference
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## **Setup**

### 1. **Create and activate a Conda environment**
```bash
conda create -n spectacles_classify python=3.9 -y
conda activate spectacles_classify
```

### 2. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## **Usage – Single Command Pipeline**

Run the **full pipeline** (dataset exploration → training → evaluation → error analysis → predictions):
```bash
python run_pipeline.py     --train_csv train.csv     --image_dir dataset     --test_csv test.csv     --test_dir dataset     --backbone resnet50     --epochs 20     --batch_size 32     --device cuda
```

This will:
1. **Explore the dataset** (class distribution, brightness, duplicates, aspect ratios).  
2. **Train the model** with two-phase fine-tuning & early stopping.  
3. **Evaluate** the model (confusion matrix, metrics, confidence histogram).  
4. **Perform error analysis** (misclassified samples + noisy label detection).  
5. **Generate predictions** for the test set (with confidence scores).  

Artifacts are stored in `outputs/` and tracked in MLflow.

---

## **API Deployment (Currently not functioning)**
Run the FastAPI inference server:
```bash
uvicorn api:app --reload
```
Send a test request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@sample.jpg"
```

---

## **Next Steps**
- **Train for more epochs** to improve accuracy.
- **Progressive unfreezing of layers** to improve training speed.
- **Include more backbones (e.g., ViT)** for a more thorough comparison.
- **Benchmark performance vs inference time** across backbones.  
- **Introduce advanced augmentation strategies** (RandAugment, Mixup, CutMix).  
- **Add a working Docker container** for production-ready deployment.  
- **Mixed precision training & gradient clipping** for faster, more stable training.  
- **Export models to ONNX, CoreML & TFLite** for mobile/edge inference.  
- **Develop a sample Android app** for on-device inference.  
