# MNIST Digit Classification with LeNet-5 and LRP

This project focuses on training a LeNet-5 convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset. Furthermore, we explore model interpretability through Layer-wise Relevance Propagation (LRP), aiming to understand the model's decision-making process.

## Project Structure

- `data_loader.py`: Contains the `MNISTDataLoader` class for loading and preprocessing the MNIST dataset.
- `lenet_model.py`: Defines the `LeNet5` class, our CNN model for digit classification.
- `model_utils.py`: Includes utility functions for training and evaluating the model, encapsulated in the `ModelUtils` class.
- `main.py`: The main script where the model training and evaluation process is executed.

## Setup and Installation

To run this project, ensure you have Python 3.6 or later installed. You will also need PyTorch and torchvision. Install the necessary libraries using pip:


```pip install torch torchvision```


## Training the Model

Run the `main.py` script to start the training process:

```python main.py```


The script will automatically download the MNIST dataset, train the LeNet-5 model, and print the loss and accuracy metrics during the training and evaluation phases.

## Understanding Model Decisions with LRP

Layer-wise Relevance Propagation (LRP) is utilized to backtrack the decision-making process of the model. This technique helps in understanding which parts of the input image most significantly influence the model's predictions.

(Note: Implementation details for LRP should be added here based on how LRP has been integrated into your project.)

## Results

After training the model, you will see the loss and accuracy printed for each epoch. The final output will include the test set's average loss and overall accuracy. Here is an example output:

```Epoch 9, Loss: 0.0035, Accuracy: 99.2%```

```Test set: Average loss: 0.0378, Accuracy: 9884/10000 (98.84%)```


## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.

## Contributors


- Your Name - [krishna200416@gmail.com](mailto:krishna200416@gmail.com)


Feel free to fork this project and contribute!


### Updates section 

map <leader>D :put =strftime('# %a %Y-%m-%d %H:%M:%S%z')<CR>


In the updated code, we introduced specialized classes, LRPLinear and LRPConv2d, to implement Layer-wise Relevance Propagation (LRP) for linear and convolutional layers, respectively. These classes are designed to replace standard PyTorch layers in the LeNet-5 architecture, enabling it to calculate and propagate relevance scores from the output back through the network, focusing specifically on the last convolutional layer. Each class captures the input during the forward pass for use in the backward pass, where the relevance propagation is customized for the layer type. For linear layers, relevance scores are calculated using a simple redistribution rule based on positive weights, and for convolutional layers, a similar approach is adapted for the convolution operation. This setup allows for an interpretation of the model's decision-making by analyzing the relevance scores of the feature maps produced by the last convolutional layer, providing insights into which features are deemed most important for the model's predictions.