import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

w3 = np.array([1, 2])
w2 = np.array([1, 3])
w1 = np.array([-2, 2])
w0 = np.array([1, -1])

def f(x):
    return x**3 @ w3 + x**2 @ w2 + x @ w1

def generate_training_data(shape):
    x = 10*np.random.random(shape) - 5
    y = f(x)
    return x, y



if __name__ == "__main__":
    x_data, y_data = generate_training_data((25000, 2))
    x_data = tf.convert_to_tensor(x_data)
    y_data = tf.convert_to_tensor(y_data)[:, np.newaxis]

    x = tf.data.Dataset.from_tensor_slices(x_data).make_one_shot_iterator().get_next()
    y = tf.data.Dataset.from_tensor_slices(y_data).make_one_shot_iterator().get_next()

    nn = tf.layers.Dense(units=1)
    y_pred = nn(x[np.newaxis, :])
    
    loss = tf.losses.mean_squared_error(labels=y[:, np.newaxis], predictions=y_pred)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    i = 0
    while True:
    #for i in range(x_data.shape[0]):
        try:
            _, loss_value = sess.run((train, loss))
            if i < 100:
                print(loss_value)
            i += 1
        except tf.errors.OutOfRangeError:
            break

    print(sess.run(y_pred))
        #x = x_data[i, :]
        #y = y_data[i]
        #actual = y
        #output = nn.forward(x)
        #loss = mse_loss(output, y)
        #gradient = mse_gradient(output, y)
        #nn.backward(gradient)

        #print("Actual: {}".format(actual))
        #print("Output: {}".format(output))
        #print("Loss: {}".format(loss))
        #print("Gradient: {}".format(gradient))

    x_test = generate_training_data((100, 2))
    y_test = f(x_test)
    #predictions = np.concatenate([nn.forward(x_test[i, :]) for i in range(x_test.shape[0])])
    ##print(np.vstack((y_test, predictions)).T)

    ##print(y_test)
    ##print(predictions)
    #print("Loss: {}".format(mse_loss(predictions, y_test)))

    #plt.scatter(y_test, predictions)
    #plt.xlabel("y_test")
    #plt.ylabel("predictions")
    #plt.show()

    #xs = np.linspace(-5, 5)
    #xs_0 = np.column_stack((xs, np.zeros(xs.shape[0])))
    #xs_1 = np.column_stack((np.zeros(xs.shape[0]), xs))
    #ys_hat_0 = f(xs_0)
    #ys_hat_1 = f(xs_1)
    #ys_0 = np.concatenate([nn.forward(xs_0[i, :]) for i in range(xs_0.shape[0])])
    #ys_1 = np.concatenate([nn.forward(xs_1[i, :]) for i in range(xs_1.shape[0])])

    #plt.subplot(121)
    #plt.plot(xs, ys_hat_0)
    #plt.plot(xs, ys_0)

    #plt.subplot(122)
    #plt.plot(xs, ys_hat_1)
    #plt.plot(xs, ys_1)

    #plt.show()
