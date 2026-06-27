import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from nutrition_data import calculate_nutrition

MODEL_PATH = "food_mobilenetv2.keras"
CLASS_NAMES_PATH = "class_names.txt"

app = FastAPI(title="SugarLens AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"Không tìm thấy class_names: {CLASS_NAMES_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]


def predict_food_from_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]

    index = int(np.argmax(predictions))
    food_key = class_names[index]
    confidence = float(predictions[index] * 100)

    top_indices = np.argsort(predictions)[::-1][:3]
    top_predictions = [
        {
            "food_key": class_names[int(i)],
            "confidence": round(float(predictions[int(i)] * 100), 2)
        }
        for i in top_indices
    ]

    return food_key, confidence, top_predictions


@app.get("/")
def home():
    return {
        "success": True,
        "message": "SugarLens AI Backend is running",
        "classes": class_names
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/analyze-food")
async def analyze_food(
    file: UploadFile = File(...),
    portion_g: int = Form(100)
):
    if portion_g <= 0:
        raise HTTPException(
            status_code=400,
            detail="Khẩu phần phải lớn hơn 0 gram."
        )

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="File gửi lên không phải ảnh hợp lệ."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi đọc ảnh: {str(e)}"
        )

    food_key, confidence, top_predictions = predict_food_from_image(image)
    nutrition = calculate_nutrition(food_key, portion_g)

    return {
        "success": nutrition is not None,
        "food_key": food_key,
        "confidence": round(confidence, 2),
        "top_predictions": top_predictions,
        "nutrition": nutrition,
        "message": None if nutrition else "Chưa có dữ liệu dinh dưỡng cho món này.",
        "medical_note": "Kết quả chỉ mang tính tham khảo, không thay thế tư vấn của bác sĩ hoặc chuyên gia dinh dưỡng."
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )