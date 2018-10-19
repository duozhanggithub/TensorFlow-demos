import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise


def add_layers(inputs, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    bias = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs, weights) + bias

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


x = tf.placeholder(tf.float32, shape=[None, 1], name='x_data')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y_data')

l1 = add_layers(x, 1, 10, activation_function=tf.nn.relu)
predicted = add_layers(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - predicted), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
#plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={x:x_data, y:y_data})
    if 1 % 50:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        #prediction_value = sess.run(predicted, feed_dict={x: x_data})
        #lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        #plt.pause(0.1)