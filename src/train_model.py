import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import keyboard
import joblib

# Загружаем данные о поворотах из файла
ROTATIONS_FILE = 'data/rotations.py'
IMG_ROTS = {}
if os.path.exists(ROTATIONS_FILE):
    try:
        with open(ROTATIONS_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            # Файл содержит строку 'rotations = {...}', исполняем ее, чтобы получить словарь
            exec(content, globals())
            IMG_ROTS = rotations # Теперь переменная rotations доступна
    except Exception as e:
        print(f"Не удалось загрузить файл с поворотами: {e}")
        IMG_ROTS = {}


print('importing things')

# Загрузка датасета
print('reading dataset')
df = pd.read_csv('data/tinder.csv')

# Проверка на наличие двух классов

ones = df[df['label'] == 1].sample(frac=1, random_state=42)
zeros = df[df['label'] == 0].sample(frac=1, random_state=42)
df = pd.concat([ones, zeros]).reset_index(drop=True)

print('read dataset')

# Путь к папке с изображениями
IMAGE_DIR = 'input'

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

# Функция для извлечения признаков
def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        
        # Применяем поворот, если он определен
        image_name = os.path.basename(image_path)
        if image_name in IMG_ROTS:
            angle = IMG_ROTS[image_name]
            if angle == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

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

print('making data')
X = []
y = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    image_path = os.path.join(IMAGE_DIR, row['filename'])
    features = extract_features(image_path)
    if features is not None:
        X.append(features)
        y.append(row['label'])

# Проверка данных
if not X:
    raise ValueError("Нет валидных данных для обучения! Проверьте файлы изображений.")

X = np.array(X)
y = np.array(y)
print('made data')

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models_dir = 'models'

# Сохраняем scaler для дальнейшего использования
scaler_path = os.path.join(models_dir, 'scaler.joblib')
joblib.dump(scaler, scaler_path)
print(f"Scaler сохранён в: {scaler_path}")


# Обучение модели
clf = HistGradientBoostingClassifier()
clf.fit(X_train, y_train)

# Сохранение модели

os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'classifier_model.joblib')
joblib.dump(clf, model_path)
print(f"Обученная модель сохранена в: {model_path}")


# Вычисление метрик
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Вывод метрик
#print(f"{roc_auc:.2f} {precision:.2f} {recall:.2f}")
print()
print("GRADIENT_HIST")
print(f"ROC-AUC: {roc_auc:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}") 

clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

# Сохранение модели

os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'classifier_model_normal.joblib')
joblib.dump(clf, model_path)
print(f"Обученная модель сохранена в: {model_path}")


# Вычисление метрик
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Вывод метрик
#print(f"{roc_auc:.2f} {precision:.2f} {recall:.2f}")
print()
print('GRADIENT_NORMAL')
print(f"ROC-AUC: {roc_auc:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}") 