from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn

# Classes
classes = ['high', 'low', 'medium']

# Load model
model = models.efficientnet_b0(pretrained=False)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(r"E:\Project2\Code\AR-DIYcode\Model_Training\efficientNet\best_oil_classifier.pth")
)

model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load image
image = Image.open(r"E:\Project2\Code\AR-DIYcode\test.jpg").convert("RGB")

image = transform(image).unsqueeze(0)

# Prediction
with torch.no_grad():

    outputs = model(image)

    predicted = torch.argmax(outputs, 1)

print("Prediction:", classes[predicted.item()])