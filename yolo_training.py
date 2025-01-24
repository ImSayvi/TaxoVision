import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.optim as optim

# Definicja datasetu YOLO
class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir  # Ścieżka do folderu z obrazami
        self.labels_dir = labels_dir  # Ścieżka do folderu z etykietami YOLO (.txt)
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
        self.transform = transform  # Transformacje obrazów
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace(".jpg", ".txt"))
        
        image = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = list(map(float, line.strip().split()))
                    class_id, x_center, y_center, width, height = data
                    labels.append(int(class_id))
                    
                    # Przekształcenie YOLO (x_center, y_center, width, height) -> (x_min, y_min, x_max, y_max)
                    x_min = (x_center - width / 2) * image.width
                    y_min = (y_center - height / 2) * image.height
                    x_max = (x_center + width / 2) * image.width
                    y_max = (y_center + height / 2) * image.height
                    boxes.append([x_min, y_min, x_max, y_max])
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
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
dataset = YoloDataset("path/to/images", "path/to/labels", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Definicja sieci neuronowej
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Inicjalizacja modelu, optymalizatora i funkcji straty
num_classes = 10  # Przykładowa liczba klas
model = SimpleCNN(num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Przykładowy trening
num_epochs = 5
for epoch in range(num_epochs):
    for images, targets in dataloader:
        labels = targets["labels"]  # Pobranie etykiet
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
