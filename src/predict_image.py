import os
import cv2
import numpy as np
import joblib
import torchvision.models as models
import torchvision.transforms as transforms
import torch


MODEL_PATH = 'models/classifier_model.joblib'
SCALER_PATH = 'models/scaler.joblib'
IMAGE_DIR = 'input'

# Загрузка обученной модели
clf = joblib.load(MODEL_PATH)

# Загрузка StandardScaler, если он был сохранён (если нет, можно обучить и сохранить отдельно)
try:
    scaler = joblib.load(SCALER_PATH)
except Exception:
    scaler = None

# Загрузка предобученной модели ResNet
model = models.resnet18(pretrained=True)
model.eval()
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# Преобразование изображений
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            features = model(img_t).numpy().flatten()
        return features
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return None

def predict_image(filename):
    image_path = filename#os.path.join(IMAGE_DIR, filename)
    features = extract_features(image_path)
    if features is None:
        raise ValueError(f"Не удалось извлечь признаки из изображения: {filename}")
    if scaler is not None:
        features = scaler.transform([features])
    else:
        features = np.array([features])
    prediction = clf.predict(features)[0]
    return prediction