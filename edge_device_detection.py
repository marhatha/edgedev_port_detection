import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import sys
from torch.utils.tensorboard import SummaryWriter

# Tensor board  default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/mnist1')


# Hyperparameters
num_epochs = 10
batch_size = 4
learning_rate = 0.001

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Load COCO dataset
train_dataset = CocoDetection(root='./dataset/ZiggoPortStatus-2/train', annFile='./dataset/ZiggoPortStatus-2/train/_annotations.coco.json', transform=transform)

test_dataset = CocoDetection(root='./dataset/ZiggoPortStatus-2/test', annFile='./dataset/ZiggoPortStatus-2/test/_annotations.coco.json', transform=transform)

# Print one image
image, annotations = train_dataset[0]  # Change the index as needed
print(f"Image shape: {image.shape}, Annotations: {annotations}")

# Get the maximum number of labels for padding
max_labels = max(len(ann) for _, ann in train_dataset)
print(f"\nmax_labels is {max_labels}")

# Load category information
with open('./dataset/ZiggoPortStatus-2/train/_annotations.coco.json', 'r') as f:
    coco_info = json.load(f)

# Get the class names
class_names = {cat['id']: cat['name'] for cat in coco_info['categories']}

# Print the class names
print("Class Names:")
for class_id, class_name in class_names.items():
    print(f"Class ID: {class_id}, Class Name: {class_name}")

# Define a collate function to handle variable number of labels
def custom_collate_fn(batch):
    images, targets = zip(*batch)
    
    # Extract category IDs as labels/classes from the target annotations
    labels = [[ann['category_id'] for ann in target] for target in targets]
#    print(f"\nlabels  is {labels}")

    # Pad labels with -1 so that each batch has the same number of labels
    padded_labels = [l + [-1]*(max_labels - len(l)) for l in labels]
#    print(f"\npadded_labels  is {padded_labels}")

    return torch.stack(images), torch.tensor(padded_labels)

# Create data loaders with custom collate function
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32*54*54, 128)  # Adjust the input size according to your resized image dimensions
        self.fc2 = nn.Linear(128, max_labels)  # Output layer with the maximum number of labels

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32*54*54)  # Adjust the input size according to your resized image dimensions
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
model = CNN()

# Tensorboard graph generation
writer.add_graph(model, torch.rand([1, 3, 224, 224]))
#writer.close()
#sys.exit()

# print weights and bias for each layers
for name, param in model.named_parameters():
    writer.add_histogram(name, param, bins='auto')

# Define loss function and optimizer
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # Forward pass
        outputs = model(images)
        
        # Convert labels to one-hot encoded tensors
        labels_one_hot = torch.zeros(outputs.shape[0], max_labels)  # Assuming max_labels is the maximum number of labels
        for batch_idx, label_batch in enumerate(labels):
            for idx, label in enumerate(label_batch):
                if label != -1:
                    labels_one_hot[batch_idx, label] = 1
 
#        print(f"\noutput shape after loop is {outputs}")
        
#        print(f"\nlabels_one_hot shape after loop is {labels_one_hot}")
 
        # Calculate loss
        loss = criterion(outputs, labels_one_hot) 

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#        if (i+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item()}')

print('Finished Training')

model.eval()            
with torch.no_grad():
    total = 0
    correct = 0           
    
    for i, (images, labels) in enumerate(test_loader):
        # Forward pass
        outputs = model(images)
        print(f"\nlogits is {outputs}")


        # Convert outputs to predicted labels using argmax
        predicted_labels = torch.argmax(outputs, dim=1)
        print(f"\npredicted_labels is {predicted_labels}")

        # Flatten the labels tensor
        labels_flat = labels.view(-1)
        print(f"\nlabels_flat is {labels_flat}")

        # Ensure labels_flat and predicted_labels have compatible shapes for comparison
        predicted_labels = predicted_labels.unsqueeze(1).expand(-1, labels_flat.size(0))
        print(f"\npredicted_labels is {predicted_labels}")

        # Calculate accuracy
        total += labels_flat.size(0)
        correct += torch.sum(labels_flat == predicted_labels.squeeze()).item()

    accuracy = correct / total
    
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
