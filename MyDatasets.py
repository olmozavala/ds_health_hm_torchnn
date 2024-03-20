# This is a torch datasets that receives the number of examples to be generated and the type of functiojn which can be
# linear, quadratic, or harmonic,.
# For linear the function is 1.5x + 0.3 + noise where noise goes between -1 and
# For quadratic the function is 2x^2 + 0.5x + 0.3 + noise where noise goes between -1 and 1
# For hamonic the function is .5x^2 + 5sin(x)  += 3cos(3x) + 2 + noise where noise goes between -1 and 1

import torch
import numpy as np
from torch.utils.data import Dataset

class SimpleFunctionsDataset(Dataset):
    def __init__(self, n_samples=100, function='linear'):
        self.n_samples = n_samples
        self.function = function

        # Generate random x values between 0 and 2π
        self.x = 2 * np.pi * torch.rand(n_samples, 1)

        # Depending on the chosen function, generate y values
        if function == 'linear':
            self.y = 1.5 * self.x + 0.3 + torch.rand(n_samples, 1) * 2 - 1  # Epsilon between -1 and 1
        elif function == 'quadratic':
            self.y = 2 * self.x.pow(2) + 0.5 * self.x + 0.3 + torch.rand(n_samples, 1) * 2 - 1
        elif function == 'harmonic':
            self.y = 0.5 * self.x.pow(2) + 5 * torch.sin(self.x) + 3 * torch.cos(3 * self.x) + 2 + torch.rand(n_samples, 1) * 2 - 1
        else:
            raise ValueError("Unsupported function. Choose from 'linear', 'quadratic', or 'harmonic'.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'x': self.x[idx], 'y': self.y[idx]}
        return sample
