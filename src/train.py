import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import MNISTModel
from datetime import datetime
import os
from tqdm import tqdm

def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Model initialization
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        
        # Calculate current accuracy and average loss
        current_accuracy = 100 * correct / total
        avg_loss = running_loss / (batch_idx + 1)
        
        # Update progress bar with loss and accuracy info
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'accuracy': f'{current_accuracy:.2f}%'
        })
    
    final_accuracy = 100 * correct / total
    print(f'\nFinal Training Accuracy: {final_accuracy:.2f}%')
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    accuracy_str = f"{final_accuracy:.2f}".replace(".", "p")
    save_path = f'models/mnist_model_{timestamp}_acc{accuracy_str}.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return save_path

if __name__ == "__main__":
    train() 