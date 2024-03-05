import pytest
from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from MyModels import DenseModel
from MyTrainings import Training  # Adjust this to the correct import for your Training function

@pytest.fixture
def simple_linear_dataset():
    # Simulate a dataset with a simple linear relationship
    x = torch.randn(100, 1)  # 100 samples, single feature
    y = 2 * x + 3  # Linear function: y = 2x + 3
    dataset = TensorDataset(x, y)
    return dataset

@pytest.fixture
def mock_dataloaders(simple_linear_dataset):
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(simple_linear_dataset))
    val_size = len(simple_linear_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(simple_linear_dataset, [train_size, val_size])

    # Create DataLoader for training and validation sets
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def test_Training(mock_dataloaders):
    train_loader, val_loader = mock_dataloaders

    # Instantiate the model and optimizer here to use in the test
    model = DenseModel(1, 1).to('cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_loss, val_loss, trained_model = Training(train_loader, val_loader, optimizer, loss=nn.MSELoss(),
                                                    model=model, epochs=1,  # Run a single epoch for testing
                                                    device='cpu')  # Use CPU for simplicity

    # Assertions to validate training behavior
    assert len(train_loss) == 1  # There should be one entry per epoch
    assert len(val_loss) == 1
    assert train_loss[0] >= 0  # Loss should be a non-negative value
    assert val_loss[0] >= 0

