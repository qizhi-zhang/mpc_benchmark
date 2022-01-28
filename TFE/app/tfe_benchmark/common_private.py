"""Provide classes to perform private training and private prediction with
logistic regression"""
import tensorflow as tf
import tf_encrypted as tfe
import os

import numpy as np



class LogisticRegression:
    """Contains methods to build and train logistic regression."""

    def __init__(self, num_features, learning_rate=0.01, l2_regularzation=1E-2):
        # self.w = tfe.define_private_variable(
        #     tf.random_uniform([num_features, 1], -0.01, 0.01))
        self.num_features=num_features
        self.w = tfe.define_private_variable(
            tf.random_uniform([num_features, 1], -1.0/np.sqrt(num_features), 1.0/np.sqrt(num_features)))
        print("self.w:", self.w)
        self.w_masked = tfe.mask(self.w)
        self.b = tfe.define_private_variable(tf.zeros([1]))
        self.b_masked = tfe.mask(self.b)
        self.learning_rate = learning_rate
        self.l2_regularzation = l2_regularzation

    @property
    def weights(self):
        """

        :return:
        """
        return self.w, self.b

    def forward(self, x, with_sigmoid=True):
        """

        :param x:
        :param with_sigmoid:
        :return:
        """
        with tf.name_scope("forward"):
            out = tfe.matmul(x, self.w_masked) + self.b_masked
            if with_sigmoid:
                y = tfe.sigmoid(out)
            else:
                y = out
            return y

    def backward(self, x, dy, learning_rate=0.01):
        """

        :param x:
        :param dy:
        :param learning_rate:
        :return:
        """
        batch_size = x.shape.as_list()[0]
        with tf.name_scope("backward"):
            dw = tfe.matmul(tfe.transpose(x), dy) / batch_size + self.l2_regularzation*self.w
            db = tfe.reduce_sum(dy, axis=0) / batch_size
            assign_ops = [
                tfe.assign(self.w, self.w - dw * learning_rate),
                tfe.assign(self.b, self.b - db * learning_rate),
            ]
            return assign_ops

    def loss_grad(self, y, y_hat):
        """

        :param y:
        :param y_hat:
        :return:
        """
        with tf.name_scope("loss-grad"):
            dy = y_hat - y
            return dy

    def fit_batch(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        with tf.name_scope("fit-batch"):
            y_hat = self.forward(x)
            dy = self.loss_grad(y, y_hat)
            fit_batch_op = self.backward(x, dy, self.learning_rate)
            return fit_batch_op

    def fit(self, sess, x, y, num_batches, progress_file):
        """

        :param sess:
        :param x:
        :param y:
        :param num_batches:
        :param progress_file:
        """
        fit_batch_op = self.fit_batch(x, y)
        with open(progress_file, "a") as f:
            for batch in range(num_batches):
                print("Batch {0: >4d}".format(batch))
                sess.run(fit_batch_op, tag='fit-batch')
                if batch % int(num_batches / 100) == 0:
                    f.write(str(1.0 * batch / num_batches) + "\n")
                    f.flush()

    def evaluate(self, sess, x, y, data_owner):
        """Return the accuracy"""

        def print_accuracy(y_hat, y) -> tf.Operation:
            """

            :param y_hat:
            :param y:
            :return:
            """
            with tf.name_scope("print-accuracy"):
                correct_prediction = tf.equal(tf.round(y_hat), y)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print_op = tf.print("Accuracy on {}:".format(data_owner.player_name),
                                    accuracy)
                print_op2 = tf.print("y_hat=:", tf.shape(y_hat))
                return (print_op, print_op2)

        with tf.name_scope("evaluate"):
            y_hat = self.forward(x)
            print_accuracy_op = tfe.define_output("YOwner",
                                                  [y_hat, y],
                                                  print_accuracy)

        sess.run(print_accuracy_op, tag='evaluate')

    def predict_batch(self, x):
        """

        :param x:
        :return:
        """
        y_hat = self.forward(x, with_sigmoid=False)
        y_hat = y_hat.reveal()
        y_hat = y_hat.to_native()
        y_hat = y_hat/tf.cast(x.shape[1],dtype='float64')
        y_hat = tf.sigmoid(y_hat)
        return y_hat

    def predict(self, sess, x, file_name, num_batches, idx, progress_file, device_name, record_num_ceil_mod_batch_size):
        """

        :param sess:
        :param x:
        :param file_name:
        :param num_batches:
        :param idx:
        :param progress_file:
        :param device_name: output device
        :param record_num_ceil_mod_batch_size:
        """

        predict_batch = self.predict_batch(x)

        with tf.device(device_name):
            predict_batch = tf.strings.as_string(predict_batch)
            predict_batch = tf.concat([idx, predict_batch], axis=1)
            predict_batch = tf.reduce_join(predict_batch, axis=1, separator=", ")
            with open(file_name, "w") as f, open(progress_file, "a") as progress_file:

                for batch in range(num_batches):
                    print("batch :", batch)
                    records = sess.run(predict_batch)

                    if batch == num_batches - 1:
                        records = records[0:record_num_ceil_mod_batch_size]
                    records = "\n".join(records.astype('str'))

                    f.write(records + "\n")

                    if batch % (1 + int(num_batches / 100)) == 0:
                        progress_file.write(str(1.0 * batch / num_batches) + "\n")
                        progress_file.flush()

    def save(self, modelFilePath, modelFileMachine="YOwner"):
        """

        :param modelFilePath:
        :param modelFileMachine:
        :return:
        """
        def _save(weights, modelFilePath) -> tf.Operation:
            weights = tf.cast(weights, "float32")
            weights = tf.serialize_tensor(weights)
            save_op = tf.write_file(modelFilePath, weights)
            return save_op

        save_ops = []
        for i in range(len(self.weights)):
            modelFilePath_i = os.path.join(modelFilePath, "param_{i}".format(i=i))
            save_op = tfe.define_output(modelFileMachine, [self.weights[i], modelFilePath_i], _save)
            save_ops = save_ops + [save_op]
        save_op = tf.group(*save_ops)
        return save_op

    def save_as_plaintext(self, modelFilePath, modelFileMachine="YOwner"):
        """

        :param modelFilePath:
        :param modelFileMachine:
        :return:
        """
        def _save(weights, modelFilePath) -> tf.Operation:
            weights = tf.cast(weights, "float32")/self.num_features
            weights = tf.strings.as_string(weights, precision=6)
            weights = tf.reduce_join(weights, separator=", ")
            save_op = tf.write_file(modelFilePath, weights)
            return save_op

        weights = tfe.concat([self.b, self.w.reshape([-1])], axis=0)

        save_op = tfe.define_output(modelFileMachine, [weights, modelFilePath], _save)
        return save_op

    def load(self, modelFilePath, modelFileMachine="YOwner"):
        @tfe.local_computation(modelFileMachine)
        def _load(param_path, shape):
            param = tf.read_file(param_path)
            param = tf.parse_tensor(param, "float32")
            param = tf.reshape(param, shape)

            print("param:", param)
            return param

        weights = []
        for i in range(len(self.weights)):
            modelFilePath_i = os.path.join(modelFilePath, "param_{i}".format(i=i))
            weights = weights + [_load(modelFilePath_i, self.weights[i].shape)]

        load_w_op = tfe.assign(self.w, weights[0])
        load_b_op = tfe.assign(self.b, weights[1])

        load_op = tf.group([load_w_op, load_b_op])
        return load_op
