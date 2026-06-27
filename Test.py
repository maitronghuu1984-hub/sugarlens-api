import tensorflow as tf
import numpy as np
from PIL import Image
from nutrition_data import calculate_nutrition

MODEL_PATH = "food_mobilenetv2.keras"
CLASS_NAMES_PATH = "class_names.txt"
IMAGE_PATH = "test5.jfif"
PORTION_G = 300

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]


def predict_food(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    index = np.argmax(predictions)
    food_key = class_names[index]
    confidence = predictions[index] * 100

    return food_key, confidence


food_key, confidence = predict_food(IMAGE_PATH)

print("===== KẾT QUẢ NHẬN DẠNG =====")
print("Món ăn:", food_key)
print("Độ tin cậy:", round(float(confidence), 2), "%")

result = calculate_nutrition(food_key, PORTION_G)

print("\n===== KẾT QUẢ DINH DƯỠNG =====")

if result is None:
    print("Chưa có dữ liệu dinh dưỡng cho món:", food_key)
else:
    print("Tên món:", result["food_name"])
    print("Khẩu phần:", result["portion_g"], "g")
    print("Carbohydrate:", result["carbohydrate_g"], "g")
    print("Đường:", result["sugar_g"], "g")
    print("Calories:", result["calories"], "kcal")
    print("Chất xơ:", result["fiber_g"], "g")
    print("Mức nguy cơ:", result["risk_level"])
    print("Khuyến nghị:", result["risk_advice"])
    print("Ghi chú món ăn:", result["food_advice"])