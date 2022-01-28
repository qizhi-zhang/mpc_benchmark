"""Private predict, submit from YOwner.
by qizhi.zqz """

import tf_encrypted as tfe
import tensorflow as tf
import json
import time
import math
import os
import platform
from commonutils.common_config import CommonConfig
from tf_encrypted.keras import backend as KE
from tfe_keeper.common_private import LogisticRegression
from tfe_keeper.read_data_tf import get_data_x, get_data_id_with_y, get_data_id_with_xy
# import sys
# from common_private import  ModelOwner, LogisticRegression, XOwner, YOwner


if platform.system() == "Darwin":
    absolute_path = "/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted"
else:
    absolute_path = "/app/file"


def run(taskId, conf, modelFileMachine, modelFilePath, progress_file, tf_config_file=None):
    """

    :param taskId:
    :param conf:
    :param modelFileMachine:
    :param modelFilePath:
    :param progress_file:
    :param tf_config_file:
    :return:
    """
    with tf.name_scope("predict"):
        trainParams = conf.get("trainParams")

        batch_size = int(trainParams.get("batchSize"))

        # dataSet=conf.get("dataSet")
        # node_id1=dataSet.get("node_id1")
        # node_id2=dataSet.get("node_id2")

        dataSet = conf.get("dataSetPredict")
        node_list = list(dataSet.keys())
        node_key_id1 = node_list.pop()
        node_key_id2 = node_list.pop()

        node_id1 = dataSet.get(node_key_id1)
        node_id2 = dataSet.get(node_key_id2)

        print("node1_containY:", node_id1.get("isContainY"))

        if node_id1.get("isContainY"):
            featureNumX = int(node_id2.get("featureNum"))
            matchColNumX = int(node_id2.get("matchColNum"))
            path_x = node_id2.get("storagePath")
            record_num = int(node_id2.get("fileRecord"))

            featureNumY = int(node_id1.get("featureNum"))
            # matchColNumY = int(node_id1.get("matchColNum"))
            path_y = node_id1.get("storagePath")
        else:
            assert node_id2.get("isContainY")
            featureNumY = int(node_id2.get("featureNum"))
            # matchColNumY = int(node_id2.get("matchColNum"))
            path_y = node_id2.get("storagePath")
            record_num = int(node_id2.get("fileRecord"))

            featureNumX = int(node_id1.get("featureNum"))
            matchColNumX = int(node_id1.get("matchColNum"))
            path_x = node_id1.get("storagePath")

        path_x = os.path.join(absolute_path, path_x)
        path_y = os.path.join(absolute_path, path_y)
        batch_num = int(math.ceil(1.0 * record_num / batch_size))
        feature_num = featureNumX + featureNumY

        CommonConfig.http_logger.info("progress_file:" + str(progress_file))

        # if len(sys.argv) >= 2:
        #   # config file was specified
        #   config_file = sys.argv[1]
        if tf_config_file:
            config = tfe.RemoteConfig.load(tf_config_file)
        else:
            # default to using local config
            config = tfe.LocalConfig([
                'XOwner',
                'YOwner',
                'RS'
            ])
        tfe.set_config(config)
        players = ['XOwner', 'YOwner', 'RS']
        prot = tfe.protocol.SecureNN(*tfe.get_config().get_players(players))
        tfe.set_protocol(prot)

        # session_target = sys.argv[2] if len(sys.argv) > 2 else None

        # tfe.set_protocol(tfe.protocol.Pond(
        #     tfe.get_config().get_player(data_owner_0.player_name), 
        #     tfe.get_config().get_player(data_owner_1.player_name)
        # ))

        @tfe.local_computation("XOwner")
        def provide_test_data_x(path):
            x = get_data_x(batch_size, path, featureNum=featureNumX, matchColNum=matchColNumX, epoch=2,
                           clip_by_value=3.0, skip_row_num=1)
            return x

        # @tfe.local_computation("YOwner")
        def provide_test_data_y(path):
            idx, y = get_data_id_with_y(batch_size, path, matchColNum=matchColNumX, epoch=2, skip_row_num=1)
            return idx, y

        # @tfe.local_computation("YOwner")
        def provide_test_data_xy(path):
            idx, x, y = get_data_id_with_xy(batch_size, path, featureNum=featureNumY, matchColNum=matchColNumX, epoch=2,
                                            clip_by_value=3.0, skip_row_num=1)
            return idx, x, y

        # batch_size, data_file, featureNum, matchColNum=2, epoch=100, clip_by_value=3.0, skip_row_num=1):

        YOwner = config.get_player("YOwner")

        if (featureNumY == 0):
            x_test = provide_test_data_x(path_x)
            with tf.device(YOwner.device_name):
                idx, y_test = provide_test_data_y(path_y)
            y_test = prot.define_private_input("YOwner", lambda: y_test)
        else:
            with tf.device(YOwner.device_name):
                idx, x_test1, y_test = provide_test_data_xy(path_y)
            x_test1 = prot.define_private_input("YOwner", lambda: x_test1)
            y_test = prot.define_private_input("YOwner", lambda: y_test)

            x_test0 = provide_test_data_x(path_x)
            x_test = prot.concat([x_test0, x_test1], axis=1)

        print("x_test:", x_test)
        print("y_test:", y_test)

        CommonConfig.http_logger.info("x_test:" + str(x_test))
        CommonConfig.http_logger.info("y_test:" + str(y_test))

        model = LogisticRegression(feature_num, learning_rate=0.1)

        CommonConfig.http_logger.info("model:" + str(model))

        load_op = model.load(modelFilePath, modelFileMachine)

        CommonConfig.http_logger.info("load_op:" + str(load_op))

        try:
            sess = KE.get_session()
            # sess.run(tfe.global_variables_initializer(), tag='init')
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())
        except Exception as e:
            CommonConfig.error_logger.exception(
                'global_variables_initializer error , exception msg:{}'.format(str(e)))
        start_time = time.time()
        CommonConfig.http_logger.info("start_time:" + str(start_time))

        print("Loading model...")
        sess.run(load_op)
        print("Load OK.")

        CommonConfig.http_logger.info("Load OK.")

        # model.fit(sess, x_train, y_train, train_batch_num)
        # model.get_KS(sess, x_test, y_test, batch_num)

        # progress_file = "./" + taskId + "/predict_progress"

        record_num_ceil_mod_batch_size = record_num % batch_size
        if record_num_ceil_mod_batch_size == 0:
            record_num_ceil_mod_batch_size = batch_size
        model.predict(sess, x_test, os.path.join(absolute_path, "tfe/{task_id}/predict".format(task_id=taskId)),
                      batch_num, idx, progress_file, YOwner.device_name, record_num_ceil_mod_batch_size)
        # model.predict(sess, x_test, os.path.join(absolute_path,
        # "tfe/{task_id}/predict".format(task_id=taskId)), batch_num, idx,
        # predict_progress_file, YOwner.device_name, record_num_ceil_mod_batch_size)

        test_time = time.time() - start_time
        print("predict_time=", test_time)

        CommonConfig.http_logger.info("predict_time=:" + str(test_time))

        with open(progress_file, "w") as f:
            f.write("1.00")
            f.flush()


if __name__ == '__main__':
    with open('./qqq/conf', 'r') as f:
        conf = f.read()
        print(conf)
    conf = conf.replace("True", "true").replace("False", "false")
    # print(input)
    conf = json.loads(conf)
    print(conf)
    progress_file = os.path.join("./qqq/train_progress")
    run(taskId="qqq", conf=conf, modelFileMachine="YOwner",
        modelFilePath="./qqq/model", progress_file=progress_file)
    run(taskId="qqq", conf=conf, modelFileMachine="YOwner",
        modelFilePath="./qqq/model", progress_file=progress_file, tf_config_file="/app/file/tfe/qqq/config.json")
