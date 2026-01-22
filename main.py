from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(title="Plant Disease Diagnosis API")

model = YOLO("yolov8_plantvillage_model.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    results = model(image)
    r = results[0]

    cls_id = int(r.probs.top1)
    confidence = float(r.probs.top1conf)
    label = model.names[cls_id]

    return {
        "disease": label,
        "confidence": round(confidence * 100, 2)
    }
