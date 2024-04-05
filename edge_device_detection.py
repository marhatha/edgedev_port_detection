import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from skimage import io
import torch.nn.functional as F

# Constants and Hyperparameters

JSON_FILE = '/Users/pmarhath/Downloads/Llama/python/chatgpt/project-12-at-2024-03-19-21-53-73daddc8/result.json'
ROOT_DIR = '/Users/pmarhath/Downloads/Llama/python/chatgpt/project-12-at-2024-03-19-21-53-73daddc8/'
NUM_EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

        # print(f"Fetching image_id: {img_info['id']}, file_name: {img_info['file_name']}")
        # print(f"Identified Categories: {sorted(categories)}")
        # print(f"Labels: {labels.numpy()}, Total Labels: {torch.sum(labels)}")

        if len(categories) != 8:
            print(f"Warning: Expected 8 categories for image_id {img_info['id']}, got {len(categories)}. Skipping.")
            return None

        return image, labels


def prepare_dataset(json_file, root_dir, train_transform, test_transform):
    full_dataset = CocoFormatDataset(json_file=json_file, root_dir=root_dir, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    test_dataset.dataset.transform = test_transform
    return train_dataset, test_dataset

def get_data_loaders(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# Define CNN model
class CNN(nn.Module):
    def __init__(self, output_size=22):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 54 * 54, 128)  # Adjust the input size according to your resized image dimensions
        self.fc2 = nn.Linear(128, output_size)  # Output layer with the maximum number of labels

    def forward(self, x): 
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 54 * 54)  # Adjust the input size according to your resized image dimensions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def initialize_model(output_size):
    model = CNN(output_size=22)
    return model

def train_model(model, train_loader, num_epochs, learning_rate, weight_decay):
    writer = SummaryWriter()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                writer.add_scalar('Training Loss', running_loss / 10, epoch * len(train_loader) + i)
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
    writer.close()

def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = torch.sigmoid(outputs) > 0.5
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    
    # Converting to numpy arrays for evaluation
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    
    return accuracy, precision, recall, f1

def visualize_image_annotations(coco_json, image_id, base_dir):
    coco = COCO(coco_json)

    # Load the image
    img = coco.loadImgs(image_id)[0]
    image_path = os.path.join(base_dir, img['file_name'])
    I = io.imread(image_path)

    # Load and display instance annotations
    plt.imshow(I); plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)

    # Print the classes
    classes = [coco.cats[ann['category_id']]['name'] for ann in anns]
    print(f'Classes: {classes}')

    plt.show()
    
def save_model(model, path="model.pth"):
    """
    Save the trained model to a file.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path="model.pth", output_size=0):
    """
    Load a model from a file for inference.
    """
    model = initialize_model(output_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def run_inference(model, image_path, transform):
    """
    Run inference on a single image and return the predicted labels.
    """
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
        predictions = torch.sigmoid(outputs) > 0.5
    return predictions.numpy()


def get_top_n_predictions(predictions, categories, n=8):
    """
    Select the top N predictions based on the model's output scores.

    Parameters:
    - predictions: The raw output scores from the model.
    - categories: The list of category names corresponding to the model's outputs.
    - n: The number of top predictions to return.

    Returns:
    - A list of the top N predicted category names.
    """
    # Squeeze predictions to remove any unnecessary dimensions
    predictions = predictions.squeeze()
    # Get the indices of the top N scores
    top_n_indices = np.argsort(predictions)[-n:]
    # Map these indices to their corresponding category names
    top_n_categories = [categories[i] for i in top_n_indices]
    
    return top_n_categories


def get_categories_from_json(json_file):
    """
    Extract category names from a COCO format JSON file.

    Parameters:
    - json_file: Path to the COCO format JSON file.

    Returns:
    - A list of category names.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    categories = {category['id']: category['name'] for category in data['categories']}
    # Sorting categories by ID to ensure they are in the correct order
    sorted_categories = [categories[cat_id] for cat_id in sorted(categories)]
    
    return sorted_categories

def select_top_predictions_per_group(predictions, categories, n=8):
    """
    Select the top prediction for each port/cable type, ensuring diverse category representation.

    Parameters:
    - predictions: The raw output scores from the model.
    - categories: The list of category names corresponding to the model's outputs.
    - n: The desired number of top predictions.

    Returns:
    - A list of the top predicted category names, respecting the port/cable groupings.
    """
    predictions = predictions.squeeze()
    # Initialize a dictionary to hold the top prediction per group
    top_predictions_per_group = {}
    
    for score, category in sorted(zip(predictions, categories), reverse=True):
        # Extract the base name (e.g., "LAN1", "WAN2") to group categories
        base_name = "_".join(category.split("_")[:-1])
        
        # Only add the top scoring prediction for each base name
        if base_name not in top_predictions_per_group:
            top_predictions_per_group[base_name] = category
            
        # Stop if we've reached the desired number of predictions
        if len(top_predictions_per_group) >= n:
            break
    
    # Extract and return the selected category names
    selected_categories = list(top_predictions_per_group.values())
    
    return selected_categories

def run_inference_raw_output(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    if transform:
        image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
    return torch.sigmoid(outputs).numpy()


print(f"\nStep1\n")
categories = get_categories_from_json(JSON_FILE)
print(categories)



print(f"\nStep2\n")
train_dataset, test_dataset = prepare_dataset(JSON_FILE, ROOT_DIR, train_transform, test_transform)
train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, BATCH_SIZE)
model = initialize_model(len(train_dataset.dataset.cat_id_to_label))

# Training
train_model(model, train_loader, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY)

# Evaluation
accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Saving the model
save_model(model, "models/v1.pth")

print('Finished Training')


print(f"\nStep3\n")

inference_model = load_model("models/v1.pth", len(train_dataset.dataset.cat_id_to_label))

# Running inference on a new image
test_image_path = "./dataset/test-images/img-1.jpg"  # Replace with the path to your new image
predictions = run_inference(inference_model, test_image_path, test_transform)
print("Inference results:", predictions)

# Using the same 'model', 'image_path', and 'transform' as before
raw_predictions = run_inference_raw_output(inference_model, test_image_path, test_transform)
print("Raw predictions:", raw_predictions)

top_predicted_categories = select_top_predictions_per_group(raw_predictions, categories, n=8)
print("Diversely Predicted Categories:", ', '.join(top_predicted_categories))


print(f"\nStep4\n")

visualize_image_annotations(JSON_FILE, 54, ROOT_DIR)
test_image_path = "./dataset/test-images/img-1.jpg"  # Replace with the path to your new image
predictions = run_inference(inference_model, test_image_path, test_transform)
print("Inference results:", predictions)

# Using the same 'model', 'image_path', and 'transform' as before
raw_predictions = run_inference_raw_output(inference_model, test_image_path, test_transform)
print("Raw predictions:", raw_predictions)

top_predicted_categories = select_top_predictions_per_group(raw_predictions, categories, n=8)
print("Diversely Predicted Categories:", ', '.join(top_predicted_categories))


print(f"\nStep5\n")

import torch
import torch.onnx
from torchvision import models
import torch.nn as nn

output_size = len(train_dataset.dataset.cat_id_to_label)
print(output_size)

model = CNN(output_size=22)

# Load the trained weights
model.load_state_dict(torch.load("models/v1.pth"))
model.eval()

# Define the dummy input as per the input size of the model. Here, it is 3x224x224 (C, H, W).
dummy_input = torch.randn(1, 3, 224, 224)

# Specify the ONNX model path
onnx_model_path = "models/model_v1.onnx"

# Export the model to the ONNX format
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11, input_names=['input'], output_names=['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})

print(f"Model has been converted to ONNX format and saved to {onnx_model_path}")

print(f"\nStep6\n")
