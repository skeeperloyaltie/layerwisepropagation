import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lrp import *

# Assuming MNISTDataLoader, LeNet5 are properly defined in their respective modules
from data_loader import MNISTDataLoader
from lenet_model import LeNet5
from model_utils import ModelUtils
import torch
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the data loader
    loader = MNISTDataLoader(batch_size=64)
    train_loader, test_loader = loader.load_data()

    # Initialize model and move it to the appropriate device
    model = LeNet5().to(device)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    model.train()
    for epoch in range(1, 11):  # Train for 10 epochs
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # Evaluation after training
    evaluate_model(model, device, test_loader)

   # After training and evaluation:
    sample_inputs, _ = next(iter(test_loader))  # Assuming you want to analyze a batch from the test set

    # Assuming you have an `apply_lrp_to_last_conv_layer` method in `lrp.py`
    from lrp import apply_lrp_to_last_conv_layer

    relevance_scores = apply_lrp_to_last_conv_layer(model, sample_inputs.to(device))
    # Now, `relevance_scores` would contain the relevance for each feature map in the last convolutional layer


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize data loader
    data_loader = MNISTDataLoader()
    train_loader, test_loader = data_loader.load_data()
    

    # Initialize and train model
    model = LeNet5()
    print(model.last_conv_layer)  # Should not be None
    print(hasattr(model.last_conv_layer, 'relprop'))  # Should be True
    model_utils = ModelUtils(model, device)
    model_utils.train(train_loader)
    model_utils.evaluate(test_loader)
    
    # Select a subset of data for LRP analysis
    images, _ = next(iter(test_loader))
    images = images.to(device)

    # Apply LRP to analyze the selected subset
    relevance_scores = apply_lrp_to_last_conv_layer(model, images)

    # Now, `relevance_scores` contains the relevance scores for the last convolutional layer's feature maps
    # You might want to visualize these scores to understand which parts of the input image were most
    # influential in the model's predictions. The exact method of visualization will depend on your
    # specific requirements and the shape of `relevance_scores`.

    # Assuming relevance_scores is the tensor you're trying to visualize
    # Sum the relevance scores across channels (dim=0 for batch, dim=1 for channels)
    relevance_summed = relevance_scores[0].sum(dim=0).detach().cpu().numpy()

    plt.imshow(relevance_summed, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title("Relevance Scores Heatmap")
    plt.savefig('relevance_scores_heatmap.png')  # Saves the plot as an image file
    plt.close()  # Closes the plot to free up resources, especially useful if generating many plots in a loop


