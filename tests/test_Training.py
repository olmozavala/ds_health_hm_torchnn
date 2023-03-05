from Training import training
from MyModels import DenseModel
import torch
import torch.nn as nn
def test_training():
    # Test the training function
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    # Test approximating a line
    y = 3*x + 5
    model = DenseModel(hidden_layers=1, neurons_per_layer=1, activation_hidden='relu', activation_output='linear')
    loss_history, trained_model = training(x=x, y=y, loss=nn.MSELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
                                       model=model, epochs=1000, device="cpu")

    print(loss_history[-1])
    assert len(loss_history) == 1000
    assert loss_history[-1] < 0.01