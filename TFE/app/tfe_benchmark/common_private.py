"""Provide classes to perform private training and private prediction with
logistic regression"""
import tensorflow as tf
import tf_encrypted as tfe
import os
#from commonutils.common_config import CommonConfig
"""
from read_data_tf import get_10w1k5col_x, get_10w1k5col_y, get_embed_op_5w_x, 
get_embed_op_5w_y, get_gaode3w_x, get_gaode3w_y
"""
import numpy as np

def fake_sigmoid(x, M=16):
  x=tfe.reveal(x).to_native()
  #M = 16
  X = np.linspace(-M, M, 256, endpoint=False)  # -M to+M的256个值
  Sigmoid = 1 / (1 + np.exp(-X))
  Sm5 = Sigmoid - 0.5

  Sm5_odd = Sm5 * 1.0
  Sm5_odd[0] = 0
  F = np.fft.fft(Sm5_odd)
  # a1 = F[1].imag
  # a2 = F[2].imag
  # a3 = F[3].imag
  # a4 = F[4].imag
  # a5 = F[5].imag
  # print(a1, a2, a3, a4, a5)
  a=F[0:6].imag
  print("a=",a)

  #a=[0, 79.01170242947703,  4.373381493344161, 21.67435635245209, 5.884090709554829,10.443694686327152]


  fake = -1.0 * (a[1] * tf.sin(2 * np.pi * (x - M) / (2 * M)) + a[2] * tf.sin(2 * 2 * np.pi * (x - M) / (2 * M))
                 + a[3] * tf.sin(2 * 3 * np.pi * (x - M) / (2 * M)) + a[4] * tf.sin(2 * 4 * np.pi * (x - M) / (2 * M)) + a[5] * tf.sin(
    2 * 5 * np.pi * (x - M) / (2 * M))) / 128
  fake=fake*5/6+0.5
  fake_sigmoid=tfe.define_public_input(player='YOwner', inputter_fn=lambda:  fake)
  return fake_sigmoid



def true_sigmoid(x):
  x=tfe.reveal(x).to_native()
  sigmoid=tf.sigmoid(x)
  true_sigmoid=tfe.define_public_input(player='YOwner', inputter_fn=lambda:  sigmoid)
  return true_sigmoid


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
                #y = fake_sigmoid(out,M=256)
                #y = true_sigmoid(out)
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

    # def get_KS(self, sess, x, y, batch_num):
    #   def print_KS(y_hat, y) -> tf.Operation:
    #     with tf.name_scope("print-KS"):
    #       # m = tf.keras.metrics.FalsePositives(list(np.array(range(1, 100))*0.01))
    #       y_hat=tf.clip_by_value(y_hat, 0.0, 1.0)
    #       FP, FP_up= tf.metrics.false_positives_at_thresholds(labels=y, predictions=y_hat, 
    #                                                           thresholds=list(np.array(range(0, 100)) * 0.01))
    #       TP, TP_up= tf.metrics.true_positives_at_thresholds(labels=y, predictions=y_hat, 
    #                                                          thresholds=list(np.array(range(0, 100)) * 0.01))
    # 
    #       FPR = FP / (tf.constant(1E-6) + FP[0])
    # 
    #       TPR = TP / (tf.constant(1E-6) + TP[0])
    #       KS= tf.reduce_max(TPR - FPR)
    #       print("KS:", KS)
    #       # m.update_state(y, y_hat)
    #       print_op=tf.print('KS=', KS )
    #       return print_op, FP_up, TP_up
    # 
    #   with tf.name_scope("get_KS"):
    #     y_hat = self.forward(x)
    # 
    #     print_KS_op = tfe.define_output("YOwner", [y_hat, y], print_KS)
    #     print("print_KS_op:", print_KS_op)
    #   sess.run(tf.local_variables_initializer())
    #   for _ in range(batch_num):
    #     sess.run(print_KS_op, tag='evaluate_KS')
    # 

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
        # sess.run(tf.local_variables_initializer())

        #CommonConfig.http_logger.info("x:" + str(x))

        predict_batch = self.predict_batch(x)

        with tf.device(device_name):
            predict_batch = tf.strings.as_string(predict_batch)
            print("idx:", idx)
            #CommonConfig.http_logger.info("idx:" + str(idx))
            predict_batch = tf.concat([idx, predict_batch], axis=1)
            predict_batch = tf.reduce_join(predict_batch, axis=1, separator=", ")
            # predict_batch=tf.reduce_join(predict_batch, separator="\n")

            #CommonConfig.http_logger.info("predict_batch:" + str(predict_batch))

            with open(file_name, "w") as f, open(progress_file, "a") as progress_file:

                for batch in range(num_batches):
                    print("batch :", batch)
                    records = sess.run(predict_batch)

                    # y_hat=str(y_hat)
                    # print(y_hat)

                    if batch == num_batches - 1:
                        records = records[0:record_num_ceil_mod_batch_size]

                    # records = str(records, encoding="utf8")
                    records = "\n".join(records.astype('str'))

                    f.write(records + "\n")

                    # if (batch % 10 == 0):
                    if batch % (1 + int(num_batches / 100)) == 0:
                        progress_file.write(str(1.0 * batch / num_batches) + "\n")
                        progress_file.flush()

        #CommonConfig.http_logger.info("predict OK")

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
