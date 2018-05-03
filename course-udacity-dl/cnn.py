import numpy as np
import tensorflow as tf


class CnnParams(object):
    def __init__(self, input_dim, conv_filter_size, conv_num_filters, conv_stride,
                 pool_size, pool_stride, num_hidden_fcn, num_classes, weights_scale):
        self.W, self.H, self.C = input_dim
        self.conv_filter_size = conv_filter_size
        self.conv_num_filters = conv_num_filters
        self.conv_stride = conv_stride
        self.pool_width = self.pool_height = pool_size
        self.pool_stride = pool_stride
        self.num_hidden_fcn = num_hidden_fcn
        self.num_classes = num_classes
        self.weights_scale = weights_scale



class CnnModel(object):
    def __init__(self, cnn_params, init_weights=True):
        self.cnn_params = cnn_params

        if init_weights:
            conv1_out_w, conv1_out_h = self.calc_conv_out_size(cnn_params.W, cnn_params.H, cnn_params.conv_filter_size, cnn_params.conv_stride)
            pool1_out_w, pool1_out_h = self.calc_pool_out_size(conv1_out_w, conv1_out_h, cnn_params.pool_width, cnn_params.pool_height, cnn_params.pool_stride)

            conv2_out_w, conv2_out_h = self.calc_conv_out_size(pool1_out_w, pool1_out_h, cnn_params.conv_filter_size, cnn_params.conv_stride)
            pool2_out_w, pool2_out_h = self.calc_pool_out_size(conv2_out_w, conv2_out_h, cnn_params.pool_width, cnn_params.pool_height, cnn_params.pool_stride)

            weights = 4 * [None]
            biases = 4 * [None]

            # First Convolutional Layer
            weights[0] = self.weight_variable([cnn_params.conv_filter_size, cnn_params.conv_filter_size, cnn_params.C, cnn_params.conv_num_filters], stddev=cnn_params.weights_scale)
            biases[0] = self.bias_variable([cnn_params.conv_num_filters], default_value=0.0)

            # Second Convolutional Layer
            weights[1] = self.weight_variable([cnn_params.conv_filter_size, cnn_params.conv_filter_size, cnn_params.conv_num_filters, cnn_params.conv_num_filters], stddev=cnn_params.weights_scale)
            biases[1] = self.bias_variable([cnn_params.conv_num_filters], default_value=1.0)

            # Hidden FullyConnectedLayer
            weights[2] = self.weight_variable([pool2_out_w * pool2_out_h * cnn_params.conv_num_filters, cnn_params.num_hidden_fcn], stddev=cnn_params.weights_scale)
            biases[2] = self.bias_variable([cnn_params.num_hidden_fcn], default_value=1.0)

            # Readout Layer
            weights[3] = self.weight_variable([cnn_params.num_hidden_fcn, cnn_params.num_classes], stddev=cnn_params.weights_scale)
            biases[3] = self.bias_variable([cnn_params.num_classes], default_value=1.0)

            self.weights = weights
            self.biases = biases
        else:
            self.weights = None
            self.biases = None

    def calc_conv_out_size(self, input_w, input_h, filter_size, conv_stride):
        pad = (filter_size - 1) / 2
        out_w = 1 + (input_w + 2 * pad - filter_size) / conv_stride
        out_h = 1 + (input_h + 2 * pad - filter_size) / conv_stride
        return (out_w, out_h)

    def calc_pool_out_size(self, input_w, input_h, pool_w, pool_h, pool_stride):
        out_w = (input_w - pool_w) / pool_stride + 1
        out_h = (input_h - pool_h) / pool_stride + 1
        return (out_w, out_h)

    def weight_variable(self, shape, stddev=0.1):
        #initial = tf.truncated_normal(shape, stddev=stddev)
        #return tf.Variable(initial)
        return np.random.normal(0, stddev, shape).astype(np.float32)

    def bias_variable(self, shape, default_value=0.1):
        #initial = tf.constant(default_value, shape=shape)
        #return tf.Variable(initial)
        return np.full(shape, default_value).astype(np.float32)



class CNN(object):
    def __init__(self, model, reg=0.0):
        self.cnn_params = model.cnn_params
        self.reg = reg

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch.
            self.X = tf.placeholder(tf.float32, shape=(None, model.cnn_params.W, model.cnn_params.H, model.cnn_params.C))
            self.Y = tf.placeholder(tf.float32, shape=(None, model.cnn_params.num_classes))

            # Using placeholder allows us to use Dropout only during Training process (not Evaluating)
            self.dropout = tf.placeholder(tf.float32)

            self.lr_start = tf.placeholder(tf.float32)
            self.lr_decay_steps = tf.placeholder(tf.float32)
            self.lr_decay_rate = tf.placeholder(tf.float32)

            self.weights = [tf.Variable(w) for w in model.weights]
            self.biases = [tf.Variable(b) for b in model.biases]

            (self.loss, self.optimizer, self.accuracy) = self.build_nn_layers()

    def update_from_model(self, model):
        with self.graph.as_default():
            self.weights = [tf.Variable(w) for w in model.weights]
            self.biases = [tf.Variable(b) for b in model.biases]

            #rebuild layers
            (self.loss, self.optimizer, self.accuracy) = self.build_nn_layers()

    def update_to_model(self, model):
        model.weights = [w.eval() for w in self.weights]
        model.biases = [b.eval() for b in self.biases]


    def build_nn_layers(self):
        # First Convolutional Layer
        conv1 = tf.nn.conv2d(self.X, self.weights[0], [1, self.cnn_params.conv_stride, self.cnn_params.conv_stride, 1], padding='SAME')
        relu1 = tf.nn.relu(conv1 + self.biases[0])
        pool1 = tf.nn.max_pool(relu1, [1, self.cnn_params.pool_width, self.cnn_params.pool_height, 1], [1, self.cnn_params.pool_stride, self.cnn_params.pool_stride, 1], padding='SAME')

        # Second Convolutional Layer
        conv2 = tf.nn.conv2d(pool1, self.weights[1], [1, self.cnn_params.conv_stride, self.cnn_params.conv_stride, 1], padding='SAME')
        relu2 = tf.nn.relu(conv2 + self.biases[1])
        pool2 = tf.nn.max_pool(relu2, [1, self.cnn_params.pool_width, self.cnn_params.pool_height, 1], [1, self.cnn_params.pool_stride, self.cnn_params.pool_stride, 1], padding='SAME')

        # Hidden FCN
        #pool2_shape = pool2.get_shape().as_list()
        pool2_shape = tf.shape(pool2)
        pool2_reshaped = tf.reshape(pool2, [pool2_shape[0], pool2_shape[1] * pool2_shape[2] * pool2_shape[3]])
        hidden_fcn = tf.nn.relu(tf.matmul(pool2_reshaped, self.weights[2]) + self.biases[2])

        # Dropout
        hidden_fcn_drop = tf.nn.dropout(hidden_fcn, self.dropout)

        # Readout Layer
        logits = tf.matmul(hidden_fcn_drop, self.weights[3]) + self.biases[3]

        # Loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.Y))

        # L2 Regularization
        if self.reg > 0:
            for w in self.weights:
                loss += self.reg * tf.nn.l2_loss(w)

        # Optimizer
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.lr_start, global_step, self.lr_decay_steps, self.lr_decay_rate, staircase=True)
        # Passing global_step to minimize() will increment it at each step.
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # Accuracy
        prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

        return (loss, optimizer, accuracy)



class CnnSolver(object):
    def __init__(self, cnn,
                 train_dataset, train_labels, valid_dataset, valid_labels,
                 batch_size=100, num_epochs=10, dropout_prob=1.0,
                 lr_start=1e-4, lr_decay_steps=100000, lr_decay_rate=0.96):
        self.cnn = cnn
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
        self.best_valid_model = CnnModel(cnn.cnn_params, init_weights=False)

    def train(self):
        num_train = self.train_dataset.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        epoch = 0

        #with tf.Session(graph=self.cnn.graph, config=tf.ConfigProto(log_device_placement=True)) as session:
        with tf.Session(graph=self.cnn.graph) as session:
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

                feed_dict = {self.cnn.X: batch_data, self.cnn.Y: batch_labels, self.cnn.dropout: self.dropout_prob,
                             self.cnn.lr_start: self.lr_start, self.cnn.lr_decay_steps:self.lr_decay_steps, self.cnn.lr_decay_rate:self.lr_decay_rate}
                # self.optimizer.run(feed_dict=feed_dict)
                _, l = session.run([self.cnn.optimizer, self.cnn.loss], feed_dict=feed_dict)

                # Check train and val accuracy on the first iteration, the last iteration, and at the end of each epoch.
                first_it = (step == 0)
                last_it = (step == num_iterations + 1)
                if first_it or last_it or epoch_end:
                    print("------------------- Epoch %d of %d ----------------------" % (epoch, self.num_epochs))
                    train_accuracy = self.cnn.accuracy.eval(feed_dict={self.cnn.X: batch_data, self.cnn.Y: batch_labels, self.cnn.dropout: 1.0})
                    print("Minibatch loss: %f" % l)
                    print("Minibatch train accuracy: %.1f%%" % (train_accuracy * 100))

                    valid_accuracy = self.cnn.accuracy.eval(feed_dict={self.cnn.X: self.valid_dataset, self.cnn.Y: self.valid_labels, self.cnn.dropout: 1.0})
                    print("Validation accuracy: %.1f%%" % (valid_accuracy * 100))

                    # Keep track of the best model
                    if valid_accuracy > self.best_valid_accuracy:
                        self.best_valid_loss = l
                        self.best_valid_accuracy = valid_accuracy
                        self.cnn.update_to_model(self.best_valid_model)

        return (self.best_valid_loss, self.best_valid_accuracy, self.best_valid_model)



