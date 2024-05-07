from collections import OrderedDict
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from datasets import load_datasets

import flwr as fl

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 50

trainloaders, testloader = load_datasets(NUM_CLIENTS, iid=True)  # Or iid=False for non-IID 

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


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


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

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader) 
        return float(loss), len(self.valloader), {
            "accuracy": float(accuracy)
        }   


def client_fn(cid) -> FlowerClient:
    net = MobileNet().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    # valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, testloader)

# Create an instance of the model and get the parameters
params = get_parameters(MobileNet())

# Pass parameters to the Strategy for server-side parameter initialization
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_evaluate_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    initial_parameters=fl.common.ndarrays_to_parameters(params),
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=10),  # Minimum of 10 rounds
    strategy=strategy,
    client_resources=None,
)