import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter

class CocoFormatDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None, required_annotations=8):
        with open(json_file, 'r') as f:
            self.coco_data = json.load(f)

        # Filter images by the number of annotations
        image_id_to_anns = Counter(ann['image_id'] for ann in self.coco_data['annotations'])
        valid_image_ids = {id for id, count in image_id_to_anns.items() if count == required_annotations}

        self.images = [img for img in self.coco_data['images'] if img['id'] in valid_image_ids]
        self.annotations = [ann for ann in self.coco_data['annotations'] if ann['image_id'] in valid_image_ids]
        self.root_dir = root_dir
        self.transform = transform
        self.cat_id_to_label = {cat['id']: idx for idx, cat in enumerate(self.coco_data['categories'])}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        ann_ids = [ann for ann in self.annotations if ann['image_id'] == img_info['id']]
        labels = torch.zeros(len(self.cat_id_to_label), dtype=torch.float32)
        categories = []

        for ann in ann_ids:
            cat_id = ann['category_id']
            label_index = self.cat_id_to_label[cat_id]
            labels[label_index] = 1
            for cat in self.coco_data['categories']:
                if cat['id'] == cat_id:
                    category_name = cat['name']
                    categories.append(category_name)
                    break

        if self.transform:
            image = self.transform(image)

        if len(categories) != 8:
            print(f"Warning: Expected 8 categories for image_id {img_info['id']}, got {len(categories)}. Skipping.")
            return None

        return image, labels
