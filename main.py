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

def visualize_relevance_scores(model, input_data):
    # Applying LRP and visualizing the relevance scores for each of the 16 feature maps
    relevance_scores = apply_lrp_to_last_conv_layer(model, input_data)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < relevance_scores.size(1):
            heatmap = relevance_scores[0, i].detach().cpu().numpy()
            im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
            ax.set_title(f'Feature Map {i+1}')
            ax.axis('off')
    plt.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal')
    plt.suptitle('Relevance Scores for 16 Feature Maps in the Last Conv Layer')
    plt.savefig('relevance_scores_feature_maps.png')
    plt.close()
    
import matplotlib.pyplot as plt
import torch

def aggregate_and_plot_relevance(relevance_scores, filename='aggregated_relevance_scores.png', bar_color='blue'):
    """
    Aggregates and plots the relevance scores for each feature map from a convolutional layer.

    Parameters:
    - relevance_scores (torch.Tensor): A tensor of relevance scores with shape (batch_size, num_feature_maps, height, width).
    - filename (str): Filename for saving the plot.
    - bar_color (str): Color of the bars in the plot.

    The function sums the relevance scores across the spatial dimensions of each feature map and plots a bar graph.
    """
    if relevance_scores.dim() != 4:
        raise ValueError("Expected relevance_scores to have 4 dimensions (batch_size, num_feature_maps, height, width)")

    # Sum over spatial dimensions to aggregate relevance scores for each feature map
    aggregated_scores = relevance_scores.sum(dim=[2, 3])  # [batch_size, num_feature_maps]

    # Ensure using only the first batch for visualization
    aggregated_scores = aggregated_scores[0].detach().cpu().numpy()  # Convert to numpy array for plotting

    # Creating the bar chart
    plt.figure(figsize=(10, 6))
    feature_map_indices = range(1, aggregated_scores.size + 1)
    plt.bar(feature_map_indices, aggregated_scores, color=bar_color)
    plt.xlabel('Feature Map')
    plt.ylabel('Aggregated Relevance Score')
    plt.title('Aggregated Relevance Scores for Feature Maps')
    plt.savefig(filename)
    plt.close()  # Close the figure to free up memory
    print(f"Plot saved as {filename}")




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
    
    
     # Visualize relevance scores
    visualize_relevance_scores(model, images)

    # agregated relavance plot
    aggregate_and_plot_relevance(relevance_scores)





