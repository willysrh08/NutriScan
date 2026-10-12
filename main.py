from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import pandas as pd
import cv2
import numpy as np
import os
from functools import lru_cache

app = Flask(__name__)

# KONFIGURASI PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
XLSX_PATH = os.path.join(BASE_DIR, "Kandungan_nutrisi.xlsx")
CLASSES_TXT = os.path.join(BASE_DIR, "classes.txt")

# LOAD MODEL & DATA 
model = YOLO(MODEL_PATH)

with open(CLASSES_TXT, "r", encoding="utf-8") as f:
    CLASS_LIST = [x.strip() for x in f if x.strip()]

nutrition_df = pd.read_excel(XLSX_PATH)
nutrition_df.set_index(nutrition_df.columns[0], inplace=True)
nutrition_df.index = nutrition_df.index.astype(str).str.lower().str.strip()

# HELPER FUNCTIONS 
@lru_cache(maxsize=256)
def get_nutrition(food_key: str) -> dict:
    """Ambil nutrisi makanan dengan caching agar cepat."""
    default = {
        "calories": 0, "carbs": 0, "protein": 0,
        "fat": 0, "fiber": 0, "sugar": 0,
        "size": "-"
    }

    key = food_key.lower().strip()
    if key not in nutrition_df.index:
        key = key.replace("_", " ")
        if key not in nutrition_df.index:
            return default

    row = nutrition_df.loc[key]
    return {
        "calories": float(row.get('Kalori', 0) or 0),
        "carbs": float(row.get('Karbohidrat', 0) or 0),
        "protein": float(row.get('Protein', 0) or 0),
        "fat": float(row.get('Lemak', 0) or 0),
        "fiber": float(row.get('Serat', 0) or 0),
        "sugar": float(row.get('Gula', 0) or 0),
        "size": str(row.get('Ukuran', '-') or '-').strip(),
    }


def read_image(file):
    """Decode image upload dengan aman."""
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Format gambar tidak valid")
    return img

# ROUTES
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image = read_image(request.files['image'])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    try:
        results = model.predict(image, imgsz=640, conf=0.25, verbose=False)
    except Exception as e:
        return jsonify({'error': f'YOLO inference failed: {e}'}), 500

    detections = []
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls.item())
        name = CLASS_LIST[cls_id]
        nutrisi = get_nutrition(name)

        detections.append({
            "name": name,
            "foodKey": name.lower().replace(" ", "_"),
            **nutrisi
        })

    return jsonify(detections)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
