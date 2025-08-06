# api.py
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
import io
import json

app = FastAPI(title="Self-Improving AI API", version="1.0")

# Load best model (update this path dynamically)
model = None
try:
    from app import genome_to_model
    # Load latest saved model
    latest_gen = 1
    import os
    for f in os.listdir("checkpoints"):
        if f.startswith("gen_") and f.endswith(".pth"):
            gen = int(f.split("_")[1].split(".")[0])
            if gen > latest_gen:
                latest_gen = gen
    agent_state = torch.load(f"checkpoints/gen_{latest_gen}.pth", map_location="cpu")
    # You’ll need to reconstruct the model from genome history
    # For now, placeholder:
    model = torch.hub.load("pytorch/vision", "resnet18", weights=None, num_classes=10)
    model.load_state_dict(agent_state)
    model.eval()
    print(f"✅ API: Loaded model gen {latest_gen}")
except Exception as e:
    print(f"⚠️ API: Using fallback model: {e}")
    model = torch.hub.load("pytorch/vision", "resnet18", weights=None, num_classes=10)
    model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image: Image.Image = Image.open(io.BytesIO(await file.read()))
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    conf, pred = torch.max(probs, 0)
    return {
        "class": class_names[pred.item()],
        "confidence": float(conf),
        "all_probs": {class_names[i]: float(probs[i]) for i in range(10)}
    }

@app.get("/")
def root():
    return {"status": "Self-Improving AI API is running", "model_generation": latest_gen}