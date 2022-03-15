#!/usr/bin/env python3
import latticex.rosetta as rtt  # difference from tensorflow
import math
import os
import csv
import tensorflow as tf
import time
import numpy as np
# from util import read_dataset

np.set_printoptions(suppress=True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(0)

EPOCHES = 15
BATCH_SIZE = 128
learning_rate = 0.01
feature_num=10
dense_dim = [feature_num, 32, 32, 1]


rtt.activate("SecureNN")
mpc_player_id = rtt.py_protocol_handler.get_party_id()

# real data
# ######################################## difference from tensorflow
# file_x = '/home/Rosetta/example/tutorials/dsets/P' + str(mpc_player_id) + "/cls_train_x.csv"
# file_y = '/home/Rosetta/example/tutorials/dsets/P' + str(mpc_player_id) + "/cls_train_y.csv"


if mpc_player_id==0:
    file_x = "/app/datasets/P0/xindai_xx_train.csv"
    file_x_test = "/app/datasets/P0/xindai_xx_test.csv"
    file_y = None
    file_y_test = None
elif mpc_player_id==1:
    file_x = "/app/datasets/P1/xindai_xy_train.csv"
    file_x_test = "/app/datasets/P1/xindai_xy_test.csv"
    file_y = "/app/datasets/P1/xindai_y_train.csv"
    file_y_test = "/app/datasets/P1/xindai_y_test.csv"
elif mpc_player_id==2:
    file_x = None
    file_x_test = None
    file_y = None
    file_y_test = None
else:
    raise Exception("mpc_player_id error")

real_X, real_Y = rtt.PrivateDataset(data_owner=(
    0, 1), label_owner=1).load_data(file_x, file_y, header=None)

real_X_test, real_Y_test = rtt.PrivateDataset(data_owner=(
    0, 1), label_owner=1).load_data(file_x_test, file_y_test, header=None)
# real_X = dataset.load_X(file_x, header=None)
#
# real_Y = dataset.load_y(file_y, header=None)
# # ######################################## difference from tensorflow
# # DIM_NUM = real_X.shape[1]
print("real_X:")
print(real_X.shape)
print("real_Y:")
print(real_Y.shape)
print("real_X_test:")
print(real_X_test.shape)
print("real_Y_test:")
print(real_Y_test.shape)

DIM_NUM = real_X.shape[1]
print(real_X)
print(real_X.shape)

X = tf.placeholder(tf.float64, [None, DIM_NUM])
Y = tf.placeholder(tf.float64, [None, 1])
print(X)
print(Y)
feature_num=DIM_NUM
# initialize W & b


z = X
for i in range(len(dense_dim)-1):
    W = tf.Variable(tf.zeros([dense_dim[i], dense_dim[i+1]], dtype=tf.float64))
    b = tf.Variable(tf.zeros(dense_dim[i+1], dtype=tf.float64))
    # print(W)
    # print(b)
    z= tf.matmul(z, W) + b
    if i !=len(dense_dim)-2:
        z = tf.nn.relu(z)


# predict
pred_Y = tf.sigmoid(z)
print(pred_Y)

# loss
logits = z
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
loss = tf.reduce_mean(loss)
print(loss)

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
print(train)

init = tf.global_variables_initializer()
print(init)

# ########### for test, reveal
# reveal_W = rtt.SecureReveal(W)
# reveal_b = rtt.SecureReveal(b)
reveal_Y = rtt.SecureReveal(pred_Y)
# ########### for test, reveal

with tf.Session() as sess:
    sess.run(init)
    #rW, rb = sess.run([reveal_W, reveal_b])
    #print("init weight:{} \nbias:{}".format(rW, rb))

    # train
    BATCHES = math.ceil(len(real_X) / BATCH_SIZE)
    start_time=time.time()
    for e in range(EPOCHES):
        for i in range(BATCHES):
            bX = real_X[(i * BATCH_SIZE): (i + 1) * BATCH_SIZE]
            bY = real_Y[(i * BATCH_SIZE): (i + 1) * BATCH_SIZE]
            sess.run(train, feed_dict={X: bX, Y: bY})

            j = e * BATCHES + i
            if j % 50 == 0 or (j == EPOCHES * BATCHES - 1 and j % 50 != 0):
                print("epoch {}, batch {}".format(e, i))
                #rW, rb = sess.run([reveal_W, reveal_b])
                #print("I,E,B:{:0>4d},{:0>4d},{:0>4d} weight:{} \nbias:{}".format(
                #    j, e, i, rW, rb))
    end_time=time.time()
    print("train time=", end_time-start_time)
    # predict
    Y_pred = sess.run(reveal_Y, feed_dict={X: real_X, Y: real_Y})
    #print("Y_pred:", Y_pred)
    with open("./predict_file", "w") as f:
            records = "\n".join(Y_pred.squeeze().astype('str'))
            # records.to_file()
            f.write(records + "\n")

print(rtt.get_perf_stats(True))
rtt.deactivate()
