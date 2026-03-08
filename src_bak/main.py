from lib.functions import ReLU, Sigmoid, Softmax, Logloss
from lib.neural_network import NeuralNetwork
import numpy as np

# load a dataset
from sklearn.datasets import load_digits

digits = load_digits()

X = digits.data
y = digits.target
y_ohe = np.array([[1 if i==value else 0 for i in range(10)] for _,value in enumerate(digits.target)]) # one hot encoding for targets

# creating neural network
nn=NeuralNetwork(layers_dimensions=[15, 10], input_dim=X.shape[1], hidden_activation=Sigmoid(), loss=Logloss(), output_activation=Softmax())

# training the model
nn.train(X, y_ohe, batch_size=10, epochs=15, learning_rate=0.001)

# show results
_=nn.get_values(display=False, chart=True, return_dict=False, include_error=True)
_=nn.plot_confusion_matrix(y, np.argmax(nn.get_values(n_epoch=-1), axis=1))