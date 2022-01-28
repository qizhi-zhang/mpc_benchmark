# -*- coding: utf-8 -*-

import tensorflow as tf
# from commonutils.common_config import CommonConfig
# read_line=tf.TextLineReader(skip_header_lines=1)
# filename_queue = tf.train.string_input_producer(["./data/10w1k5col_x.csv"])
# recodes=read_line.read(filename_queue)
# 
# recodes=tf.decode_csv(recodes, [[0.2]]*290, field_delim=",")


def get_data_xy(batch_size, data_file, featureNum, matchColNum=2, epoch=100, clip_by_value=3.0, skip_row_num=1):
    def line_split(r):
        return tf.decode_csv(r, [["a"]] * matchColNum + [[0.2]] * featureNum + [[1]], field_delim=",")

    def norm(x):
        x = tf.cast(x, tf.float32)
        return tf.clip_by_value(x, -clip_by_value, clip_by_value)

    data = tf.data.TextLineDataset(data_file).skip(skip_row_num).map(line_split)
    # .shuffle(buffer_size=50000, seed=10086)

    batch_data_iter_x = data.map(lambda *r: tf.stack(r[matchColNum:-1], axis=-1)).map(norm)\
        .repeat(epoch).batch(batch_size).make_one_shot_iterator()
    batch_data_iter_y = data.map(lambda *r: r[-1]).repeat(epoch).\
        batch(batch_size).make_one_shot_iterator()

    batch_data_x = batch_data_iter_x.get_next()
    batch_data_x = tf.reshape(batch_data_x, shape=[batch_size, featureNum])

    batch_data_y = batch_data_iter_y.get_next()
    batch_data_y = tf.reshape(batch_data_y, shape=[batch_size, 1])

    return (batch_data_x, batch_data_y)


def get_data_id_with_y(batch_size, data_y_file, matchColNum=2, epoch=100, skip_row_num=1):

    def line_split(r):
        return tf.decode_csv(r, [["a"]] * matchColNum + [[0.9]], field_delim=",")

    data = tf.data.TextLineDataset(data_y_file).skip(skip_row_num).map(
        line_split)
    # .shuffle(buffer_size=50000, seed=10086)

    batch_data_iter = data.map(lambda *r: r[matchColNum]).repeat(epoch).batch(
        batch_size)
    #  .make_one_shot_iterator()
    print("batch_data_iter:", batch_data_iter)
    # batch_data = batch_data_iter.get_next()
    batch_data = tf.compat.v1.data.make_one_shot_iterator(batch_data_iter).get_next()
    batch_data = tf.reshape(batch_data, shape=[batch_size, 1])
    print("batch_data:", batch_data)

    batch_idx_iter = data.map(lambda *r: tf.stack(r[0:matchColNum], axis=-1)).repeat(epoch).batch(
        batch_size).make_one_shot_iterator()
    batch_idx = batch_idx_iter.get_next()
    batch_idx = tf.reshape(batch_idx, shape=[batch_size, matchColNum])
    print("batch_idx:", batch_idx)
    return (batch_idx, batch_data)


def get_data_id_with_xy(batch_size, data_file, featureNum, matchColNum=2, epoch=100, clip_by_value=3.0, skip_row_num=1):
    def line_split(r):
        return tf.decode_csv(r, [["a"]] * matchColNum + [[0.2]] * featureNum + [[1]], field_delim=",")

    def norm(x):
        x = tf.cast(x, tf.float32)
        return tf.clip_by_value(x, -clip_by_value, clip_by_value)

    data = tf.data.TextLineDataset(data_file).skip(skip_row_num)\
        .map(line_split)  # .shuffle(buffer_size=50000, seed=10086)

    batch_data_iter_x = data.map(lambda *r: tf.stack(r[matchColNum:-1], axis=-1)).map(norm).repeat(epoch).batch(
        batch_size).make_one_shot_iterator()
    batch_data_iter_y = data.map(lambda *r: r[-1]).repeat(epoch).batch(
        batch_size).make_one_shot_iterator()
    batch_idx_iter = data.map(lambda *r: tf.stack(r[0:matchColNum], axis=-1)).repeat(epoch).batch(
        batch_size).make_one_shot_iterator()

    batch_data_x = batch_data_iter_x.get_next()
    batch_data_x = tf.reshape(batch_data_x, shape=[batch_size, featureNum])

    batch_data_y = batch_data_iter_y.get_next()
    batch_data_y = tf.reshape(batch_data_y, shape=[batch_size, 1])

    batch_idx = batch_idx_iter.get_next()
    batch_idx = tf.reshape(batch_idx, shape=[batch_size, matchColNum])

    return (batch_idx, batch_data_x, batch_data_y)


def get_data_x(batch_size, data_x_file, featureNum, matchColNum=2, epoch=100, clip_by_value=3.0, skip_row_num=1):
    def line_split(r):
        return tf.decode_csv(r, [["a"]] * matchColNum + [[0.2]] * featureNum, field_delim=",")

    def norm(x):
        x = tf.cast(x, tf.float32)
        return tf.clip_by_value(x, -clip_by_value, clip_by_value)

    data = tf.data.TextLineDataset(data_x_file).skip(skip_row_num).map(line_split)
    # .shuffle(buffer_size=50000, seed=10086)

    batch_data_iter = data.map(lambda *r: tf.stack(r[matchColNum:], axis=-1)).map(norm)\
        .repeat(epoch).batch(batch_size).make_one_shot_iterator()

    batch_data = batch_data_iter.get_next()
    return tf.reshape(batch_data, shape=[batch_size, featureNum])


def get_data_y(batch_size, data_y_file, matchColNum=2, epoch=100, skip_row_num=1):

    def line_split(r):
        return tf.decode_csv(r, [["a"]] * matchColNum + [[0.9]], field_delim=",")

    data = tf.data.TextLineDataset(data_y_file).skip(skip_row_num)\
        .map(line_split)
    # .shuffle(buffer_size=50000, seed=10086)

    batch_data_iter = data.map(lambda *r: r[matchColNum]).repeat(epoch)\
        .batch(batch_size)
    #  .make_one_shot_iterator()

    print("batch_data_iter:", batch_data_iter)

    # batch_data = batch_data_iter.get_next()
    batch_data = tf.compat.v1.data.make_one_shot_iterator(batch_data_iter).get_next()
    print("batch_data:", batch_data)

    return tf.reshape(batch_data, shape=[batch_size, 1])


if __name__ == '__main__':
    file = "/Users/qizhi.zqz/projects/TFE/tf-encrypted/examples/test_on_morse_datas/data/embed_op_fea_5w_format_x.csv"
    q = get_data_x(64, file,
                   291, matchColNum=2, epoch=100, clip_by_value=3.0, skip_row_num=1)
    print(q)
