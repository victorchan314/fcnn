import numpy as np
import matplotlib.pyplot as plt

#w3 = np.array([1, 2])
#w2 = np.array([1, 3])
#w1 = np.array([-2, 2])
#w0 = np.array([1, -1])

w3 = np.array([1])
w2 = np.array([-2])
w1 = np.array([1])

def f(x):
    return x**3 @ w3 + x**2 @ w2 + x @ w1



def generate_training_data(shape):
    return 10*np.random.random(shape) - 10

def mse_loss(y, y_hat):
    return np.mean(np.power(y - y_hat, 2))

def mse_gradient(y, y_hat):
    return 2*y - 2*y_hat

leaky_relu = lambda x: np.maximum(-0.1*x, x)

leaky_relu_gradient = lambda x: np.greater_equal(x, 0).astype("float64") * 1.1 - 0.1



class Cell:
    def __init__(self, input_shape):
        pass

    def forward(self, *values):
        raise NotImplementedError

    def backward(self, *values):
        raise NotImplementedError

class LinearCell(Cell):
    def __init__(self, input_shape, alpha=0.01, activation_fn=lambda x: np.maximum(0, x), dfn=lambda x: np.greater_equal(x, 0).astype("int64")):
        self.x = 10*np.random.random(input_shape) - 5
        self.w = 10*np.random.random(input_shape) - 5
        self.b = 0
        self.y = 0
        self.a = alpha
        self.fn = activation_fn
        self.dfn = dfn

    def forward(self, x):
        self.x = x
        self.y = x @ self.w + self.b
        return self.fn(self.y)

    def backward(self, gradient):
        dl_df = gradient
        df_dy = self.dfn(self.y)
        dy_dx = self.w
        dy_dw = self.x
        dy_db = 1

        dl_dx = dl_df * df_dy * dy_dx
        dl_dw = dl_df * df_dy * dy_dw
        dl_db = dl_df * df_dy * dy_db

        self.w -= self.a * dl_dw
        self.b -= self.a * dl_db

        return dl_dx

class LinearLayer:
    def __init__(self, input_size, output_size, alpha=0.01):
        self.input_shape = (input_size,)
        self.cells = [LinearCell(self.input_shape, alpha, activation_fn=leaky_relu, dfn=leaky_relu_gradient) for _ in range(output_size)]
        self.gradient = np.zeros(self.input_shape)

    def forward(self, x):
        output = np.array([cell.forward(x) for cell in self.cells])

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
    x_data = generate_training_data((25000, 1))
    y_data = f(x_data)
    nn = NeuralNetwork(1, 1, [3] * 5, alpha=1e-8)
    for i in range(x_data.shape[0]):
        x = x_data[i, :]
        y = y_data[i]
        actual = y
        output = nn.forward(x)
        loss = mse_loss(output, y)
        gradient = mse_gradient(output, y)
        nn.backward(gradient)

        #print("Actual: {}".format(actual))
        #print("Output: {}".format(output))
        #print("Loss: {}".format(loss))
        #print("Gradient: {}".format(gradient))

    x_test = generate_training_data((100, 1))
    y_test = f(x_test)
    predictions = np.concatenate([nn.forward(x_test[i, :]) for i in range(x_test.shape[0])])
    #print(np.vstack((y_test, predictions)).T)

    #print(y_test)
    #print(predictions)
    print("Loss: {}".format(mse_loss(predictions, y_test)))

    plt.scatter(y_test, predictions)
    plt.xlabel("y_test")
    plt.ylabel("predictions")
    plt.show()

    xs = np.linspace(-5, 5)[:, np.newaxis]
    #xs_0 = np.column_stack((xs, np.zeros(xs.shape[0])))
    #xs_1 = np.column_stack((np.zeros(xs.shape[0]), xs))
    #ys_hat_0 = f(xs_0)
    #ys_hat_1 = f(xs_1)
    #ys_0 = np.concatenate([nn.forward(xs_0[i, :]) for i in range(xs_0.shape[0])])
    #ys_1 = np.concatenate([nn.forward(xs_1[i, :]) for i in range(xs_1.shape[0])])

    ys_hat = f(xs)
    ys = np.concatenate([nn.forward(xs[i, :]) for i in range(xs.shape[0])])
    plt.plot(xs, ys_hat)
    plt.plot(xs, ys)

    #plt.subplot(121)
    #plt.plot(xs, ys_hat_0)
    #plt.plot(xs, ys_0)

    #plt.subplot(122)
    #plt.plot(xs, ys_hat_1)
    #plt.plot(xs, ys_1)

    plt.show()
