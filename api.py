import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from model import get_model
import mlflow
import time
from datetime import datetime
import uuid
import os

app = FastAPI()

# Load best model
MODEL_PATH = "outputs/final_mobilenetv3.pth"
model = get_model(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.get("/")
def health_check():
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    timestamp = datetime.utcnow().isoformat()

    try:
        image = Image.open(file.file).convert('RGB')
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze().tolist()
        pred = torch.argmax(outputs, dim=1).item()
    response_time = time.time() - start_time

    # Log request details to MLflow
    mlflow.set_experiment("ClassificationAPI")
    with mlflow.start_run(nested=True):
        mlflow.log_param("request_id", request_id)
        mlflow.log_param("file_name", file.filename)
        mlflow.log_param("file_size", len(await file.read()))
        mlflow.log_param("timestamp", timestamp)
        mlflow.log_metric("predicted_class", pred)
        mlflow.log_metric("response_time_sec", response_time)
        for idx, p in enumerate(probs):
            mlflow.log_metric(f"class_{idx}_prob", p)

    return {
        "request_id": request_id,
        "prediction": int(pred),
        "confidence_scores": probs,
        "response_time_sec": response_time,
        "timestamp": timestamp
    }