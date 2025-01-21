import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class AnimalDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            labels (dict): Dictionary mapping image filenames to labels.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.labels = labels
        self.image_files = list(labels.keys())
        self.transform = transform

    def __len__(self):
        """Returns the number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, index):
        """Fetches the image and label at the given index."""
        img_name = self.image_files[index]
        img_path = os.path.join(self.image_dir, img_name)
        label = self.labels[img_name]

        # Open the image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, label

# Example usage:
# dataset = AnimalDataset("images/", {"cat1.jpg": 0, "dog1.jpg": 1}, transform=None)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
