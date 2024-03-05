import unittest
import torch
from math import isclose
from MyDatasets import SimpleFunctionsDataset

class TestSimpleFunctionsDataset(unittest.TestCase):
    def test_linear_function(self):
        dataset = SimpleFunctionsDataset(n_samples=100, function='linear')
        self.assertEqual(len(dataset), 100)

        x, y = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape, torch.Size([1]))
        self.assertEqual(y.shape, torch.Size([1]))

        # Test if the generated data follows the linear function
        x = dataset.x
        expected_y = 1.5 * x + 0.3
        expected_y = (expected_y - expected_y.mean()) / expected_y.std()
        self.assertTrue(isclose(y[0].item(), expected_y[0].item(), rel_tol=1))

    def test_quadratic_function(self):
        dataset = SimpleFunctionsDataset(n_samples=100, function='quadratic')
        self.assertEqual(len(dataset), 100)

        x, y = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape, torch.Size([1]))
        self.assertEqual(y.shape, torch.Size([1]))

        # Test if the generated data follows the quadratic function
        x = dataset.x
        expected_y = 2 * x**2 + 0.5 * x + 0.3
        expected_y = (expected_y - expected_y.mean()) / expected_y.std()
        self.assertTrue(isclose(y[0].item(), expected_y[0].item(), rel_tol=1))

    def test_harmonic_function(self):
        dataset = SimpleFunctionsDataset(n_samples=100, function='harmonic')
        self.assertEqual(len(dataset), 100)

        x, y = dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.shape, torch.Size([1]))
        self.assertEqual(y.shape, torch.Size([1]))

        # Test if the generated data follows the harmonic function
        x = dataset.x
        expected_y = .5 * x**2 + 5 * torch.sin(x) + 3 * torch.cos(3 * x) + 2
        expected_y = (expected_y - expected_y.mean()) / expected_y.std()
        self.assertTrue(isclose(y[0].item(), expected_y[0].item(), rel_tol=1))

if __name__ == '__main__':
    unittest.main()