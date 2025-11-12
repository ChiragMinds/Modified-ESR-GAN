# src/esrgan/data.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.ToTensor()
        self.file_list = self._extract_files()

    def _extract_files(self):
        file_list = []
        low_quality_path = os.path.join(self.root_dir, 'low_quality')
        high_quality_path = os.path.join(self.root_dir, 'high_quality')
        for filename in sorted(os.listdir(low_quality_path)):
            low = os.path.join(low_quality_path, filename)
            high = os.path.join(high_quality_path, filename)
            if os.path.isfile(low) and os.path.isfile(high):
                file_list.append((low, high))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        low_img_path, high_img_path = self.file_list[idx]
        low_image = Image.open(low_img_path).convert('L')
        high_image = Image.open(high_img_path).convert('L')
        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)
        return low_image, high_image

def get_loaders(train_dir, val_dir, batch_size=8, img_size=(64,64), num_workers=2):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    train_ds = CustomDataset(train_dir, transform)
    val_ds = CustomDataset(val_dir, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
