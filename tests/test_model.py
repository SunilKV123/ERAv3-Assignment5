import torch
import pytest
from src.model import MNISTModel
from torchvision import datasets, transforms
import os

def test_model_parameters():
    model = MNISTModel()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 100000, f"Model has {total_params} parameters, should be < 100000"

def test_input_output_shape():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_model_accuracy():
    # Load the latest model
    model = MNISTModel()
    models_dir = 'models'
    latest_model = max([os.path.join(models_dir, f) for f in os.listdir(models_dir)])
    model.load_state_dict(torch.load(latest_model, map_location=torch.device('cpu')))
    
    # Test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Accuracy is {accuracy}%, should be > 80%" 