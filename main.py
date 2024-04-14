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

    relevance_scores_flattened = relevance_summed.flatten()

    plt.hist(relevance_scores_flattened, bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Relevance Scores')
    plt.xlabel('Relevance Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('relevance_histogram.png')  # Save the histogram
    plt.close()

    epochs = range(1, len(model_utils.train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, model_utils.train_losses, 'b-', label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, model_utils.train_accuracies, 'r-', label='Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.savefig('training_metrics.png')  # Save the figure to a file
    plt.close()  # Close the plot
    
   




