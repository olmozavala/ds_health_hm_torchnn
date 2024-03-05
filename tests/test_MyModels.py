import pytest
import torch
import torch.nn as nn
from MyModels import DenseModel

@pytest.fixture
def dense_model():
    return DenseModel()


def test_dense_model_creation():
    dense_model = DenseModel()
    assert isinstance(dense_model, nn.Module)


@pytest.mark.parametrize("hidden_layers, neurons_per_layer", [
    (1, 10),
    (2, 20),
    (3, 5),
])
def test_forward_pass(hidden_layers, neurons_per_layer):
    model = DenseModel(hidden_layers, neurons_per_layer)
    x = torch.randn(10, 1)  # Batch size of 10
    output = model(x)
    assert output.shape == torch.Size([10, 1]), "Output shape should be [10, 10] for any input configuration"

@pytest.mark.parametrize("hidden_layers, neurons_per_layer, activation_hidden, activation_output", [
    (2, 2, 'relu', 'linear'),
    (2, 20, 'sigmoid', 'tanh'),
    (3, 5, 'tanh', 'sigmoid'),
])
def test_model_construction(hidden_layers, neurons_per_layer, activation_hidden, activation_output):
    model = DenseModel(hidden_layers, neurons_per_layer, activation_hidden, activation_output)
    # Assuming each layer has a bias term, adjust the expected_params calculation if not
    expected_params = (neurons_per_layer + neurons_per_layer)  +\
                        neurons_per_layer**2 * (hidden_layers) + neurons_per_layer * hidden_layers +\
                        (neurons_per_layer + 1)  # Last layer to output
    if hidden_layers == 0:  # Adjust for models with only one layer
        expected_params = 2  # One weight and one bias
    total_params = sum(p.numel() for p in model.parameters())

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    assert total_params == expected_params, f"Expected {expected_params} parameters, got {total_params}"
    # Add more checks here if necessary, such as checking for the correct activation functions

def test_single_hidden_layer():
    dense_model = DenseModel(hidden_layers=1)
    input_tensor = torch.randn(10, 1)  # Batch size 10, input size 1
    output = dense_model(input_tensor)
    assert output.shape == (10, 1)  # Output shape should match input shape