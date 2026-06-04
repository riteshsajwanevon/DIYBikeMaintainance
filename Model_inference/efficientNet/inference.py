from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms, models
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Classes
classes = ['high', 'low', 'medium']

# Load model
model = models.efficientnet_b0(pretrained=False)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(BASE_DIR / "Model_Training" / "efficientNet" / "best_oil_classifier.pth")
)

model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load image
image = Image.open(BASE_DIR / "test.jpg").convert("RGB")

image = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():

    outputs = model(image)

    predicted = torch.argmax(outputs, 1)

print("Prediction:", classes[predicted.item()])