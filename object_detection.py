import os
import torch
import torchvision.transforms as T
import torchvision.models.detection as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json

class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir  # Ścieżka do folderu z obrazami
        self.labels_dir = labels_dir  # Ścieżka do folderu z etykietami (JSON)
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
        self.transform = transform  # Transformacje obrazów
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.image_files[idx].replace(".jpg", ".json"))
        
        image = Image.open(img_path).convert("RGB")
        with open(label_path, 'r') as f:
            labels = json.load(f)  # Wczytaj etykiety
        
        boxes = torch.tensor(labels["boxes"], dtype=torch.float32)  # Współrzędne obiektów
        labels = torch.tensor(labels["labels"], dtype=torch.int64)  # Klasy obiektów
        
        target = {"boxes": boxes, "labels": labels}
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

# Przykładowe transformacje
transform = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
])

# Tworzenie zbioru danych
dataset = CustomDataset("path/to/images", "path/to/labels", transform=transform)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Pobranie przykładowej próbki
data_iter = iter(dataloader)
images, targets = next(data_iter)

# Sprawdzenie rozmiarów
print(images.shape)  # (batch_size, 3, 224, 224)
print(targets)  # Lista słowników z 'boxes' i 'labels'

# Wybór modelu detekcji obiektów
model = models.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Ustawienie modelu w tryb ewaluacji (bez trenowania)

# Przetwarzanie batcha przez model
with torch.no_grad():
    predictions = model(images)

# Wyświetlenie wyników
for i, pred in enumerate(predictions):
    print(f"Obraz {i}: Detekcja obiektów")
    print(pred["boxes"])  # Współrzędne wykrytych obiektów
    print(pred["labels"])  # Klasy wykrytych obiektów
    print(pred["scores"])  # Prawdopodobieństwa
