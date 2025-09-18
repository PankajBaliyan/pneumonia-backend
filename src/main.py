import base64
import io

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse

app = FastAPI(title="Pneumonia Detection API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://pneumoai.pankajdev.in",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Label", "X-Probability"],
)

# Model configuration
MODEL_PATH = "models/resnet_pneumonia.h5"
XRAY_MODEL_PATH = "models/xray_filter.h5"
LAST_CONV_LAYER = "conv5_block3_out"


# ---------------- Utility Functions ----------------
def load_model():
    """Load trained model from file."""
    return tf.keras.models.load_model(MODEL_PATH)

def load_xray_filter_model():
    """Load trained model from file."""
    global xray_filter_model
    xray_filter_model = tf.keras.models.load_model(XRAY_MODEL_PATH)


def preprocess_pil_image(pil_img, target_size=(224, 224)):
    """Resize and normalize PIL image for model input."""
    pil_img = pil_img.convert("RGB").resize(target_size)
    arr = np.array(pil_img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap for model predictions."""
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
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())

    return (heatmap / max_val).numpy()

def overlay_heatmap_on_image(original_img_cv2, heatmap, alpha=0.4):
    """Overlay heatmap on original image (OpenCV BGR format)."""
    heatmap_resized = cv2.resize(heatmap, (original_img_cv2.shape[1], original_img_cv2.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(original_img_cv2, 1 - alpha, heatmap_color, alpha, 0)

def cv2_to_base64_png(cv2_img):
    """Convert OpenCV image to base64-encoded PNG."""
    _, buffer = cv2.imencode(".png", cv2_img)
    return base64.b64encode(buffer).decode("utf-8")

# ---------------- Startup ----------------
@app.on_event("startup")
def startup_event():
    """Load and warmup model at application startup."""
    global model
    model = load_model()
    load_xray_filter_model()
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    try:
        _ = model.predict(dummy)
    except Exception as e:
        print("Model warmup failed:", e)


# ---------------- API Endpoints ----------------
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Pneumonia Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict pneumonia presence from chest X-ray image."""
    if file.content_type.split("/")[0] != "image":
        return JSONResponse(
            status_code=400,
            content={
                "data": None,
                "status": 0,
                "message": "File must be an image."
            }
        )

    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents))

    orig_cv2 = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_array = preprocess_pil_image(pil_img)

    # Step 1: Check if it’s an X-ray
    xray_pred = xray_filter_model.predict(img_array)[0][0]
    if xray_pred < 0.5:   # threshold
        return JSONResponse(
            status_code=400,
            content={
                "data": None,
                "status": 0,
                "message": "Not a chest X-ray."
            }
        )

    # Step 2: Run pneumonia prediction
    preds = model.predict(img_array)
    prob = float(preds[0][0]) if preds.shape[-1] == 1 else float(preds[0][1])
    label = "Pneumonia" if prob >= 0.5 else "Normal"

    try:
        heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER)
        overlay = overlay_heatmap_on_image(orig_cv2, heatmap)

        _, buffer = cv2.imencode(".png", overlay)
        img_bytes = io.BytesIO(buffer.tobytes())

        headers = {"X-Label": label, "X-Probability": str(prob)}
        return StreamingResponse(img_bytes, media_type="image/png", headers=headers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM error: {e}")