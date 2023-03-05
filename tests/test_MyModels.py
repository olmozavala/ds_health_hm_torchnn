from MyModels import DenseModel
import torch.nn as nn
import torch

def test_DenseModel():

    # Test the DenseModel class
    # Simple linear model with 1 hidden layer
    model = DenseModel(hidden_layers=0, neurons_per_layer=1, activation_hidden='relu', activation_output='linear')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params == 4

    model = DenseModel(hidden_layers=1, neurons_per_layer=1, activation_hidden='relu', activation_output='relu')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params == 6

    model = DenseModel(hidden_layers=5, neurons_per_layer=10, activation_hidden='sigmoid', activation_output='linear')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params == 581

    # Check the model is the correct type
    assert isinstance(model, DenseModel)
    assert isinstance(model, nn.Module)