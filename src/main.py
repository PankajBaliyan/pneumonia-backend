import io
import cv2
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
from starlette.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="Pneumonia Detection API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173", "https://pneumoai.pankajdev.in"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Label", "X-Probability"]
)

# Define: model path, last convolutional layer name
MODEL_PATH = "models/resnet_pneumonia.h5"
LAST_CONV_LAYER = "conv5_block3_out"  # if you used ResNet50 earlier

# --- Utility functions ---
def load_model():
    # prefer SavedModel; tf.keras.models.load_model works with .h5 and SavedModel
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def preprocess_pil_image(pil_img, target_size=(224,224)):
    pil_img = pil_img.convert("RGB").resize(target_size)
    arr = np.array(pil_img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1, H, W, 3)
    return arr

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]  # shape (h, w, channels)

    # Weighted combination of activation maps
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    # ReLU & normalize
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())
    heatmap /= max_val
    return heatmap.numpy()

def overlay_heatmap_on_image(original_img_cv2, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_img_cv2.shape[1], original_img_cv2.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original_img_cv2, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed

def cv2_to_base64_png(cv2_img):
    _, buffer = cv2.imencode(".png", cv2_img)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return b64

# --- Load model at startup ---
@app.on_event("startup")
def startup_event():
    global model
    model = load_model()
    # Warmup with a zero input to reduce first-request latency
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    try:
        _ = model.predict(dummy)
    except Exception as e:
        print("Model warmup failed:", e)

# --- Prediction endpoint ---
@app.get("/")
async def root():
    return {"message": "Pneumonia Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents))

    # keep a copy of original for overlay (use cv2 BGR)
    orig_cv2 = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_array = preprocess_pil_image(pil_img)

    # Prediction
    preds = model.predict(img_array)
    prob = float(preds[0][0]) if preds.shape[-1] == 1 else float(preds[0][1])
    label = "Pneumonia" if prob >= 0.5 else "Normal"

    # Grad-CAM
    try:
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
        overlay = overlay_heatmap_on_image(orig_cv2, heatmap)

        # Encode overlay as PNG in memory
        _, buffer = cv2.imencode(".png", overlay)
        img_bytes = io.BytesIO(buffer.tobytes())

        # Send both metadata + image back using headers
        headers = {
            "X-Label": label,
            "X-Probability": str(prob)
        }

        return StreamingResponse(img_bytes, media_type="image/png", headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM error: {e}")