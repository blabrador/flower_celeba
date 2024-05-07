from collections import OrderedDict
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from datasets import load_datasets

DEVICE = torch.device("cuda:0")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__}"
)

NUM_CLIENTS = 1

trainloaders, testloader = load_datasets(NUM_CLIENTS, iid=True, full_dataset_debug=True)  # Or iid=False for non-IID 

class MobileNet(nn.Module):
    def __init__(self) -> None:
        super(MobileNet, self).__init__()
        # Load pre-trained MobileNetV2 and freeze feature extractor
        self.mobilenet = mobilenet_v2(pretrained=True)
        for param in self.mobilenet.features.parameters():
            param.requires_grad = False
        
        num_features = self.mobilenet.classifier[-1].in_features
        num_classes = 1 # Binary classification
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4), # TODO
            nn.Linear(num_features, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mobilenet.features(x)  # Extract features with MobileNetV2
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)) # From torchvision mobilnetv2
        x = torch.flatten(x, 1)
        x = self.classifier(x)  # New classifier
        return x


def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels, demographics in trainloader:
            images, labels = images.to(DEVICE), labels.float().to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (outputs.data.round() == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    correct, total, loss = 0, 0, 0.0
    demographics_correct = {}  
    demographics_total = {} 
    net.eval()
    with torch.no_grad():
        for images, labels, demographics in testloader:
            images, labels = images.to(DEVICE), labels.float().to(DEVICE).unsqueeze(1)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            predicted = outputs.data.round()
            total += labels.size(0)

    loss /= len(testloader.dataset)
    accuracy = (predicted == labels).sum().item() / total

    return loss, accuracy


net = MobileNet().to(DEVICE)
EPOCHS=20

train(net, trainloaders[0], epochs=EPOCHS)

# Evaluation
loss, accuracy = test(net, testloader)
print(f"Overall Results - Loss: {loss}, Accuracy: {accuracy}")