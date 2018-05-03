import numpy as np
import tensorflow as tf


class MultiLayerNet(object):
    # {affine - relu - dropout} x (L - 1) - affine - softmax

    def __init__(self, input_dim, hidden_dims, num_classes, reg=0.0, weights_scale=0.1, weights=None, biases=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
            self.X = tf.placeholder(tf.float32, shape=(None, input_dim))
            self.Y = tf.placeholder(tf.float32, shape=(None, num_classes))

            # Using placeholder allow us to use Dropout only during Training process (not Evaluating)
            self.dropout = tf.placeholder(tf.float32)

            self.lr_start = tf.placeholder(tf.float32)
            self.lr_decay_steps = tf.placeholder(tf.float32)
            self.lr_decay_rate = tf.placeholder(tf.float32)

            if weights == None:
                self.weights, self.biases = self.buildNNParams(input_dim, hidden_dims, num_classes, weights_scale)
            else:
                self.weights = [tf.Variable(w) for w in weights]
                self.biases = [tf.Variable(b) for b in biases]

            (self.loss, self.optimizer, self.accuracy) = self.buildNNOperations(self.X, self.Y, self.dropout,
                                                                                self.weights, self.biases,
                                                                                hidden_dims, reg,
                                                                                self.lr_start, self.lr_decay_steps, self.lr_decay_rate)

    def buildNNParams(self, input_dim, hidden_dims, num_classes, weights_scale):
        weights = []
        biases = []

        hidden_dims = hidden_dims[:]
        hidden_dims.append(num_classes)

        prev_hidden_dim = input_dim
        for hidden_dim in hidden_dims:
            weights.append(tf.Variable(tf.truncated_normal([prev_hidden_dim, hidden_dim], stddev=weights_scale), trainable=True))
            biases.append(tf.Variable(tf.constant(0.1, shape=[hidden_dim]), trainable=True))

            prev_hidden_dim = hidden_dim

        return (weights, biases)

    def buildNNOperations(self, X, Y, dropout_prob, weights, biases, hidden_dims, reg, lr_start, lr_decay_steps, lr_decay_rate):
        weights_reg_sum = 0
        dropout = None
        ind = 0
        for hidden_dim in hidden_dims:
            if dropout == None:
                act = tf.matmul(X, weights[ind]) + biases[ind]
            else:
                act = tf.matmul(dropout, weights[ind]) + biases[ind]

            # RELu
            relu = tf.nn.relu(act)

            # Dropout
            dropout = tf.nn.dropout(relu, dropout_prob)

            # Collect L2 Reg
            if reg > 0:
                weights_reg_sum += tf.nn.l2_loss(weights[ind])

            ind += 1

        # Loss
        logits = tf.matmul(dropout, weights[ind]) + biases[ind]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, Y))

        # L2 Regularization
        weights_reg_sum += tf.nn.l2_loss(weights[ind])
        loss += reg * weights_reg_sum

        # Optimizer
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(lr_start, global_step, lr_decay_steps, lr_decay_rate, staircase=True)
        # Passing global_step to minimize() will increment it at each step.
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Accuracy
        prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return (loss, optimizer, accuracy)




class Solver(object):
    def __init__(self, model,
                 train_dataset, train_labels, valid_dataset, valid_labels,  # test_dataset, test_labels,
                 batch_size=100, num_epochs=10, dropout_prob=1.0,
                 lr_start=1e-4, lr_decay_steps=100000, lr_decay_rate=0.96):
        self.model = model
        self.train_dataset = train_dataset
        self.train_labels = train_labels
        self.valid_dataset = valid_dataset
        self.valid_labels = valid_labels
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_prob = dropout_prob
        self.lr_start = lr_start
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate

        self.best_valid_loss = None
        self.best_valid_accuracy = None
        self.best_valid_weights = None
        self.best_valid_biases = None

    def train(self):
        num_train = self.train_dataset.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        epoch = 0

        #with tf.Session(graph=self.model.graph, config=tf.ConfigProto(log_device_placement=True)) as session:
	with tf.Session(graph=self.model.graph) as session:
            session.run(tf.initialize_all_variables())
            print("Initialized")

            for step in xrange(num_iterations):
                epoch_end = (step + 1) % iterations_per_epoch == 0
                if epoch_end:
                    epoch += 1

                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
                offset = (step * self.batch_size) % (num_train - self.batch_size)
                # Generate a minibatch.
                batch_data = self.train_dataset[offset:(offset + self.batch_size), :]
                batch_labels = self.train_labels[offset:(offset + self.batch_size), :]

                feed_dict = {self.model.X: batch_data, self.model.Y: batch_labels, self.model.dropout: self.dropout_prob,
                             self.model.lr_start: self.lr_start, self.model.lr_decay_steps:self.lr_decay_steps, self.model.lr_decay_rate:self.lr_decay_rate}
                # self.optimizer.run(feed_dict=feed_dict)
                _, l = session.run([self.model.optimizer, self.model.loss], feed_dict=feed_dict)

                # Check train and val accuracy on the first iteration, the last iteration, and at the end of each epoch.
                first_it = (step == 0)
                last_it = (step == num_iterations + 1)
                if first_it or last_it or epoch_end:
                    print("------------------- Epoch %d of %d ----------------------" % (epoch, self.num_epochs))
                    train_accuracy = self.model.accuracy.eval(
                        feed_dict={self.model.X: batch_data, self.model.Y: batch_labels, self.model.dropout: 1.0})
                    print("Minibatch loss: %f" % l)
                    print("Minibatch train accuracy: %.1f%%" % (train_accuracy * 100))

                    valid_accuracy = self.model.accuracy.eval(
                        feed_dict={self.model.X: self.valid_dataset, self.model.Y: self.valid_labels, self.model.dropout: 1.0})
                    print("Validation accuracy: %.1f%%" % (valid_accuracy * 100))

                    # Keep track of the best model
                    if valid_accuracy > self.best_valid_accuracy:
                        self.best_valid_loss = l
                        self.best_valid_accuracy = valid_accuracy
                        self.best_valid_weights = [w.eval() for w in self.model.weights]
                        self.best_valid_biases = [b.eval() for b in self.model.biases]

        return (self.best_valid_loss, self.best_valid_accuracy, self.best_valid_weights, self.best_valid_biases)



