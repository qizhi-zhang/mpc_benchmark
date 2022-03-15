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
feature_num=291
dense_dim=[feature_num, 32, 32, 1]


rtt.activate("SecureNN")
mpc_player_id = rtt.py_protocol_handler.get_party_id()

# real data
# ######################################## difference from tensorflow
# file_x = '/home/Rosetta/example/tutorials/dsets/P' + str(mpc_player_id) + "/cls_train_x.csv"
# file_y = '/home/Rosetta/example/tutorials/dsets/P' + str(mpc_player_id) + "/cls_train_y.csv"
file_x = "/app/datasets/P{}/embed_op_fea_5w_format_x_train.csv".format(str(mpc_player_id))
file_y = "/app/datasets/P{}/embed_op_fea_5w_format_y_train.csv".format(str(mpc_player_id))

print("file_x=", file_x)
print("file_y=", file_y)

dataset = rtt.PrivateDataset(data_owner=(0,), label_owner=1)



real_X = dataset.load_X(file_x, header=None)

real_Y = dataset.load_y(file_y, header=None)

file_x_test = "/app/datasets/P{}/embed_op_fea_5w_format_x_test.csv".format(str(mpc_player_id))
file_y_test = "/app/datasets/P{}/embed_op_fea_5w_format_y_test.csv".format(str(mpc_player_id))

print("file_x_test=", file_x_test)
print("file_y_test=", file_y_test)

dataset_test = rtt.PrivateDataset(data_owner=(0,), label_owner=1)



real_X_test = dataset_test.load_X(file_x_test, header=None)

real_Y_test = dataset_test.load_y(file_y_test, header=None)

# # ######################################## difference from tensorflow
# # DIM_NUM = real_X.shape[1]
print("real_X_test:")
print(real_X_test.shape)
print("real_Y_test:")
print(real_Y_test.shape)








X = tf.placeholder(tf.float64, [None, feature_num])
Y = tf.placeholder(tf.float64, [None, 1])
#_, X = tf.split(id_X, num_or_size_splits=[match_num, feature_num], axis=1)
# _, Y = tf.split(id_Y, num_or_size_splits=[match_num, 1], axis=1)
X = tf.clip_by_value(X, clip_value_max=5.0, clip_value_min=-5.0)
Y = tf.clip_by_value(Y, clip_value_max=5.0, clip_value_min=-5.0)

print(X)
print(Y)

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
    pred_batch_size = 32
    pred_BATCHES = math.ceil(len(real_Y_test) / BATCH_SIZE)

    #print("Y_pred:", Y_pred)
    with open("./predict_file", "w") as f:
        for i in range(pred_BATCHES):
            Y_pred = sess.run(reveal_Y, feed_dict={X: real_X_test[(i * pred_batch_size): (i + 1) * pred_batch_size], Y: real_Y_test[(i * pred_batch_size): (i + 1) * pred_batch_size]})
            records = "\n".join(Y_pred.squeeze().astype('str'))
            # records.to_file()
            f.write(records + "\n")

print(rtt.get_perf_stats(True))
rtt.deactivate()
