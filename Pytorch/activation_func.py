import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x) 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))     

def arctan(x):
    return np.arctan(x)

x = np.random.randn(100) 
print(x)       
y = relu(x)

plt.scatter(x, y)
plt.title('relu function')
plt.show()

y = sigmoid(x)

plt.scatter(x, y)
plt.title('sigmoid function')
plt.show()

y = arctan(x)

plt.scatter(x, y)
plt.title("arctan function")
plt.show()