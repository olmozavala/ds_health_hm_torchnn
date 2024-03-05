#  PyTorch NN Basics (40 Pts)

The objective of this homework is to apply some of the tools we have learned so far (widgets, GitHub, PyTorch, and NN) 
to build a test bed to evaluate multiple NN parameters easily. 

## Test bed for Simple Neural Networks

We will make a test bed where we can create **dense** NNs with different parameters. 
The parameters we will be able to test in real time are:
1. Number of hidden layers.
2. Number of neurons for the hidden layers.
3. Activation function used in hidden layers (Sigmoid, Tanh, or ReLu)
4. Activation function used in the output layer (Sigmoid, Tanh, ReLu, Linear)
5. Number of epochs to train the model.

## 1 Make a *dataset* to create random samples from different functions (10) 
Inside **MyDatasets.py** create a class called **SimpleFunctionsDataset** that
receives the following parameters in the constructor:
* **n_samples**. Number of samples to be generated. (default value 100)
* **function**. The function to be approximated. Options: 
    * 'linear' → $y = 1.5x + 0.3 + \epsilon$. Epsilon represent uniform distribution error between -1 and 1
    * 'quadratic' → $y = 2x^2 + 0.5x + 0.3 + \epsilon$. Epsilon represent uniform distribution error between -1 and 1
    * 'harmonic' → $y = .5x^2 + 5\sin(x) + 3\cos(3x) + 2 + \epsilon$. Epsilon represent uniform distribution error between -1 and 1.

The values of x should be generated randomly between 0 and $2\pi$. 

This dataset should be a subclass of `torch.utils.data.Dataset` and 
contain the required methods to be used in a `torch.utils.data.DataLoader` (e.g. `__len__` and `__getitem__`).

**IMPORTANT** Normalize your output data to be mean 0 and standard deviation 1. 

## 2 Make a *dynamic* NN module (10) 
Inside **MyModels.py** create a class called **DenseModel** that
imports from **nn.Module** and receives the following parameters in the constructor:

* **hidden_layers**. Number of hidden layers (default value 1)
* **neurons_per_layer**. Number of neurons for hidden layers (default value 1)
* **activation_hidden**. Activation function to be used in the hidden layers. Options: 'relu', 'sigmoid', 'tanh','linear'. 
Default value of 'relu'.
* **activation_output**. Activation function to be used in the output layer. 
Options: 'relu', 'sigmoid', 'tanh', 'linear'. 

Depending on the input values, the constructor should create the appropriate number of layers, neurons, and activation functions. 

Finally, create a **forward** method that receives the input data and returns the output of the previously created NN model.

## 3 Make a *dynamic* Training module (10) 
Inside **MyTraining.py** create a function called **Training** that
receives the following parameters:

* **train_dataloader**. The dataloader to be used in the training.
* **validation_dataloader**. The dataloader to be used in the training.
* **optimizer**. The optimizer to be used in the training. 
* **loss**. The loss function to be used in the training. (default value: `nn.MSELoss()`)
* **model**. The model to be trained. (default value: `DenseModel()`)
* **epochs**. Number of epochs to train the model. (default value: `500`)
* **batch_size**. The batch size to be used in the training. (default value: `10`)


The function should train the model for the specified number of epochs and return:
1. The training loss function for each epoch as a list.
2. The validation loss function for each epoch as a list.
3. The trained model.

## 4 Make your test bed (10)
Following the provided jupyter notebook called **TestBed.py** fill the missing code of the widget
to make a test bed where you can:

1. A dropdown to select the number of hidden layers. Options from 1 to 5.
2. A dropdown to select the number of neurons for each hidden layer. Options from 1 to 100.
3. A dropdown to select the number of epochs to train the model. Options from 1 to 1000 every 100.
4. A dropdown menu to select the activation function for the hidden layers. Options are: 'relu', 'sigmoid', 'tanh', 'linear'. (default value: 'relu')
5. A dropdown menu to select the activation function for the output layer. Options are: 'relu', 'sigmoid', 'tanh', 'linear'. (default value: 'linear')
6. A dropdown menu to select the function to be approximated. Options are: 'linear', 'quadratic', 'harmonic'. (default value: 'linear')
7. A button to start the training.

The widget should show the following plots: 
1. Figure 1. Observations to be approximated and the function approximated by the trained model.
2. Figure 2. The training and validation loss function or each epoch.

Example:
![](./images/ExampleOutputN.png)

## 5 Neural network analysis (10)
Please identify a set of parameters that show the following:

1. **Overfitting** (5) Can you show an example of overfitting? What parameters did you use? why do you think it is overfitting? Show a plot of your training and validation loss that corresponds to the overfitting example.
2. **Underfitting** (5) Can you show an example of underfitting? What parameters did you use? why do you think it is underfitting? Show a plot of your training and validation loss that corresponds to the underfitting example. 

## 6 Extra points batch normalization (10)
Inside **MyModels.py** create a class called **DenseModelBN** that
includes a parameter for batch normalization. If the parameter is true then include BN after each hidden layer.

Additionally, update your widget to include a checkbox to include batch normalization and call this updated widget **TestBedBN.py**.

Include an example where BN helps to improve the accuracy of a deep model (with multiple hidden layers).