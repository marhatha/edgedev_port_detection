import torch.nn as nn
import torch.nn.functional as F

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
