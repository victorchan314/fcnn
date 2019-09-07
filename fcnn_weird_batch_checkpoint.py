import numpy as np
import matplotlib.pyplot as plt

w3 = np.array([1, 2])
w2 = np.array([1, 3])
w1 = np.array([-2, 2])
w0 = np.array([1, -1])

def f(x):
    return x**3 @ w3 + x**2 @ w2 + x @ w1



def mse_loss(y, y_hat):
    return np.mean(np.power(y - y_hat, 2))

def mse_gradient(y, y_hat):
    return 2*y - 2*y_hat

class Cell:
    def __init__(self, input_shape):
        pass

    def forward(self, *values):
        raise NotImplementedError

    def backward(self, *values):
        raise NotImplementedError

class LinearCell(Cell):
    def __init__(self, input_shape, alpha=0.01, activation_fn=lambda x: np.maximum(0, x), dfn=lambda x: np.greater_equal(x, 0).astype("int64")):
        self.x = np.random.random(input_shape)
        self.w = np.random.random(input_shape)
        self.b = 0
        self.y = 0
        self.a = alpha
        self.fn = activation_fn
        self.dfn = dfn

    def forward(self, x):
        self.x = x
        self.y = x @ self.w + self.b
        if len(self.y.shape) == 0:
            self.y = np.array([self.y])

        return self.fn(self.y)

    def backward(self, gradient):
        #print(self.w)
        #print(self.b)
        dl_df = gradient
        df_dy = self.dfn(self.y)
        dy_dx = self.w
        dy_dw = self.x
        dy_db = 1

        dl_dx = ((df_dy[:, np.newaxis] @ dy_dx[:, np.newaxis].T).T * dl_df).T
        dl_dw = ((dl_df * df_dy).T * dy_dw).T
        dl_db = dl_df * df_dy * dy_db

        self.w -= self.a * dl_dw
        self.b -= self.a * dl_db
        #print(self.w)
        #print(self.b)

        return dl_dx

class LinearLayer:
    def __init__(self, input_size, output_size, alpha=0.01):
        self.input_shape = (input_size,)
        self.cells = [LinearCell(self.input_shape, alpha) for _ in range(output_size)]
        self.gradient = np.zeros(self.input_shape)

    def forward(self, x):
        if len(x.shape) > 1:
            output = np.column_stack((cell.forward(x) for cell in self.cells))
        else:
            output = np.concatenate([cell.forward(x) for cell in self.cells])

        return output

    def backward(self, gradient):
        self.gradient = np.zeros(self.input_shape)
        for i, cell in enumerate(self.cells):
            self.gradient += cell.backward(gradient[i])

        return self.gradient

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layer_sizes = [], alpha=0.01):
        self.layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        self.layers = [LinearLayer(self.layer_sizes[i], self.layer_sizes[i+1], alpha) for i in range(len(self.layer_sizes) - 1)]

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)



if __name__ == "__main__":
    x_data = 10*np.random.random((1000, 2))
    y_data = f(x_data)
    nn = NeuralNetwork(2, 1, [5], alpha=0.01)
    for i in range(100):
        x = x_data[10*i:10*i + 10, :]
        y = y_data[10*i:10*i+10]
        actual = y
        output = nn.forward(x)
        loss = mse_loss(output, y)
        gradient = mse_gradient(output, y)
        nn.backward(gradient)

        print("Actual: {}".format(actual))
        print("Output: {}".format(output))
        print("Loss: {}".format(loss))
        #print("Gradient: {}".format(gradient))

    x_test = 10*np.random.random((100, 2))
    y_test = f(x_test)
    predictions = np.array([nn.forward(x_test[i, :]) for i in range(100)])

    print(y_test)
    print(predictions)
    print(np.mean(np.power(y_test - predictions, 2)))

    plt.scatter(y_test, predictions)
    plt.show()
