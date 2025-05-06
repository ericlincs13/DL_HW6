import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset as torchData
from torchvision import transforms
import json


class ObjectConverter():
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.object_dic: dict[str, int] = json.load(f)
            self.class_num = len(self.object_dic)

    def convert(self, label: list[str]):
        onehot_label = np.zeros(self.class_num, dtype=np.float32)
        for l in label:
            onehot_label[self.object_dic[l]] = 1
        return onehot_label


class TrainingDataset(torchData):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.image_dir = os.path.join(self.root, 'images')
        self.filenames = os.listdir(self.image_dir)

        self.object_converter = ObjectConverter(
            os.path.join(self.root, 'objects.json'))

        with open(os.path.join(self.root, 'train.json'), 'r') as f:
            labels: dict[str, list[str]] = json.load(f)

        self.labels = {
            filename: self.object_converter.convert(labels[filename])
            for filename in labels
        }

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        path = os.path.join(self.image_dir, filename)
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        label = self.labels[filename]

        return image, label


class TestingDataset(torchData):
    def __init__(self, root, filename='test.json'):
        super().__init__()
        self.root = root

        self.object_converter = ObjectConverter(
            os.path.join(self.root, 'objects.json'))

        with open(os.path.join(self.root, filename), 'r') as f:
            labels: list[list[str]] = json.load(f)

        self.labels = [
            self.object_converter.convert(label) for label in labels
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index]
