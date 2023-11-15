from flask import Flask, request, send_file
import cv2
import os
import numpy as np
import tensorflow as tf
from huggingface_hub import snapshot_download
from PIL import Image
import requests
import io


app = Flask(__name__)

def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image

def resize_image(image, target_size=(1024, 1024)):
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def download_image(url):
    response = requests.get(url, stream=True).raw
    image = Image.open(response)
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def preprocess_image(image):
    image = resize_crop(image)
    image = image.astype(np.float32) / 127.5 - 1
    image = np.expand_dims(image, axis=0)
    image = tf.constant(image)
    return image

# Load the model and extract concrete function.
model_path = snapshot_download("sayakpaul/whitebox-cartoonizer")
loaded_model = tf.saved_model.load(model_path)
concrete_func = loaded_model.signatures["serving_default"]

@app.route("/")
def hello():
    return "Hello"

@app.route("/get-image", methods=["POST"])
def get_image():
    try:
        # Get image URL from the request
        # image_url = "https://github.com/Mi-TZ/ahaha/blob/main/gdgdg.jpg?raw=true"

        image_file = request.files.get("image")
        if not image_file:
            return "No image file provided."
        
        # Download and preprocess image.
        # image = download_image(image_url)
        image = Image.open(image_file)
        image = image.convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        resized_image = resize_image(image)


        preprocessed_image = preprocess_image(resized_image)

        # Run inference.
        result = concrete_func(preprocessed_image)["final_output:0"]

        # Post-process the result and convert it to bytes.
        output = (result[0].numpy() + 1.0) * 127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output_image = Image.fromarray(output)
        
        # Convert the image to bytes
        img_byte_array = io.BytesIO()
        output_image.save(img_byte_array, format="PNG")
        img_byte_array = img_byte_array.getvalue()

        # Return the image as a Flask response
        return send_file(io.BytesIO(img_byte_array), mimetype="image/png")

    except Exception as e:
        print("Error:", e)
        return "lmao"

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 33507))
    app.run(host='0.0.0.0', port=port)


# // web: gunicorn main:app -t 100 --keep-alive 100
