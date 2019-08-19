import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

input_size = 28*28
n_layers = 10
n_classes = 10
n_units = 30
batch_size = 50
epochs = 400

data_size = 5000

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

# X = np.random.rand(data_size, input_size)
# y = np.random.randint(0, n_classes, [data_size])

X = x_train[:data_size]
y = y_train[:data_size]

y_onehot = np.zeros([data_size, n_classes])
y_onehot_test = np.zeros([len(y_test), n_classes])

for a in range(data_size):
	y_onehot[a][y[a]] = 1
for a in range(len(y_test)):
	y_onehot_test[a][y_test[a]] = 1

inputs = tf.placeholder(tf.float32, [None, input_size])
original_input = inputs
labels = tf.placeholder(tf.float32, [None, n_classes])

for a in range(n_layers):
	layer = tf.layers.dense(inputs, n_units, 'selu', name='layer'+str(a))
	inputs = tf.concat([inputs, layer], 1, name='concatenation'+str(a))

logits = tf.layers.dense(inputs, n_classes, None, name='output')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
predictions = tf.nn.softmax(logits)

optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

with tf.Session() as session:
	tf.initialize_all_variables().run()
	print("Initialized")

	for a in range(epochs):
		_, l, p = session.run([optimizer, loss, predictions], feed_dict={original_input:X, labels:y_onehot})

		if a%50==0:
			print("Epoch:", a+1, "\tLoss:", l, "\tAccuracy:", accuracy(p, y_onehot))

	test_p = session.run(predictions, feed_dict={original_input:x_test, labels:y_onehot_test})

	print("Test Accuracy: ", accuracy(test_p, y_onehot_test))

	
	tf.summary.FileWriter('./logs', session.graph)
	# print("Output: ", res)
