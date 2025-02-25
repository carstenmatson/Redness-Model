import os
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import io
import segmentation_models_pytorch as smp

# ✅ Lazy Model Loading for Lower Memory Usage
def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return torch.jit.script(model)  # Convert to TorchScript for efficiency

# ✅ Ensure Model Path Works Locally & on Render
MODEL_PATH = os.getenv("MODEL_PATH", "skin_redness.pth")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

model = None  # Defer model loading until first request

# ✅ Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ Initialize Flask App
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "🔥 Skin Redness Detection API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        model = load_model()  # Load model only on first request

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image).sigmoid()  # Apply sigmoid activation

        prediction = output.squeeze().tolist()  # Convert tensor to list
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Ensure Port Works on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)



