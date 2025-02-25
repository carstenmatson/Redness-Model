import os
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from PIL import Image
import io
import segmentation_models_pytorch as smp

# ✅ Define Model with the Same Architecture
def load_model(model_path):
    model = smp.Unet(
        encoder_name="resnet34",  # Must match the training architecture
        encoder_weights=None,  # Do NOT load pretrained weights
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))  # Load weights
    model.eval()  # Set to evaluation mode
    return model

# ✅ Ensure Model Path Works Locally & on Render
MODEL_PATH = os.getenv("MODEL_PATH", "skin_redness.pth")  # Use env var or local file
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

# ✅ Load Model (with TorchScript Optimization)
model = load_model(MODEL_PATH)
model = torch.jit.script(model)  # Convert to TorchScript for lower RAM usage

# ✅ Define Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Match model input size
    transforms.ToTensor(),
])

# ✅ Initialize Flask App
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "🔥 Skin Redness Detection API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image).sigmoid()  # Apply sigmoid to get probability

        prediction = output.squeeze().tolist()  # Convert tensor to list
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Ensure Port Works on Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 for local testing
    app.run(host="0.0.0.0", port=port)



