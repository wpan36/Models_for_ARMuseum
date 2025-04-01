from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# 加载模型
model = tf.keras.models.load_model("best_model.keras")

# 预处理函数（与你训练模型时的预处理一致）
def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))  # 调整大小
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array / 127.5) - 1  # 归一化
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image'].read()
    input_tensor = preprocess(image)

    predictions = model.predict(input_tensor)[0].tolist()
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
