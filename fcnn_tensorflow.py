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
    #x = (2*np.random.random(shape) + 3) * (np.random.randint(2, size=shape) * 2 - 1)
    y = f(x)
    return x, y



if __name__ == "__main__":
    x_data, y_data = generate_training_data((25000, 2))

    x = tf.placeholder(tf.float32, (1, x_data.shape[1]))
    y = tf.placeholder(tf.float32, (1, 1))

    nn = x

    for i in range(3):
        nn = tf.layers.dense(nn, units=10, activation=tf.nn.relu)

    nn = tf.layers.dense(nn, units=1)
    y_pred = nn
    
    loss = tf.losses.mean_squared_error(labels=y, predictions=y_pred)

    optimizer = tf.train.GradientDescentOptimizer(1e-6)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for i in range(x_data.shape[0]):
        _, loss_value = sess.run((train, loss), feed_dict={x: x_data[i, :][np.newaxis, :], y: np.array([y_data[i]])[:, np.newaxis]})
        #print(loss_value)

    x_test, y_test = generate_training_data((100, 2))

    predictions = np.concatenate([sess.run(y_pred, feed_dict={x: x_test[i, :][np.newaxis, :]})[0] for i in range(x_test.shape[0])])
    #print(np.vstack((y_test, predictions)).T)

    print("Loss: {}".format(sess.run(tf.losses.mean_squared_error(labels=y_test, predictions=predictions))))

    plt.scatter(y_test, predictions)
    plt.xlabel("y_test")
    plt.ylabel("predictions")
    plt.show()

    xs = np.linspace(-5, 5)
    xs_0 = np.column_stack((xs, np.zeros(xs.shape[0])))
    xs_1 = np.column_stack((np.zeros(xs.shape[0]), xs))
    ys_hat_0 = f(xs_0)
    ys_hat_1 = f(xs_1)
    ys_0 = np.concatenate([sess.run(y_pred, feed_dict={x: xs_0[i, :][np.newaxis, :]})[0] for i in range(xs_0.shape[0])])
    ys_1 = np.concatenate([sess.run(y_pred, feed_dict={x: xs_1[i, :][np.newaxis, :]})[0] for i in range(xs_1.shape[0])])

    plt.subplot(121)
    plt.plot(xs, ys_hat_0)
    plt.plot(xs, ys_0)

    plt.subplot(122)
    plt.plot(xs, ys_hat_1)
    plt.plot(xs, ys_1)

    plt.show()
