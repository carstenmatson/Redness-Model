from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
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

# ✅ Load Model
MODEL_PATH = r"C:\Users\cmats\Documents\project\Model Folder\skin_redness.pth"
model = load_model(MODEL_PATH)

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
    
    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image).sigmoid()  # Apply sigmoid to get probability

    prediction = output.squeeze().tolist()  # Convert tensor to list

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



