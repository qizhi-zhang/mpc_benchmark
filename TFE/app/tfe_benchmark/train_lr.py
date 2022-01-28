"""Private training on combined data from several data owners"""
import tf_encrypted as tfe
import json
# from common_private import  ModelOwner, LogisticRegression, XOwner, YOwner
from tfe_keeper.common_private import LogisticRegression
from tfe_keeper.read_data_tf import get_data_xy, get_data_x, get_data_y
from tf_encrypted.keras import backend as KE
import tensorflow as tf
# import sys
import time
import platform
import os
from commonutils.common_config import CommonConfig

if platform.system() == "Darwin":
    absolute_path = "/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted"
else:
    absolute_path = "/app/file"


def run(taskId, conf, modelFileMachine, modelFilePath, modelFilePlainTextPath, tf_config_file=None):

    progress_file = os.path.join(absolute_path, "tfe/" + taskId + "/train_progress")
    CommonConfig.http_logger.info("progress_file:" + str(progress_file))
    with open(progress_file, "w") as f:
        f.write(str(0.0) + "\n")
        f.flush()

    trainParams = conf.get("trainParams")

    CommonConfig.http_logger.info("train_lr/run:  trainParams:" + str(trainParams))

    learningRate = float(trainParams.get("learningRate"))
    batch_size = int(trainParams.get("batchSize"))
    epoch_num = int(trainParams.get("maxIter"))
    # epsilon = float(trainParams.get("epsilon"))
    # regularizationL1=float(trainParams.get("regularizationL1"))
    # regularizationL2=float(trainParams.get("regularizationL2"))

    dataSet = conf.get("dataSet")

    CommonConfig.http_logger.info("dataSet:" + str(dataSet))

    try:
        node_list = list(dataSet.keys())
        node_key_id1 = node_list.pop()
        node_key_id2 = node_list.pop()

        node_id1 = dataSet.get(node_key_id1)
        node_id2 = dataSet.get(node_key_id2)
    except Exception as e:
        CommonConfig.error_logger.exception(
            'get node from dataSet {} error, exception msg:{}'.format(str(dataSet), str(e)))

    # node_id1=dataSet.get("node_id1")
    # node_id2=dataSet.get("node_id2")

    # print("node1_containY:", node_id1.get("isContainY"))
    CommonConfig.http_logger.info("node1_containY:" + str(node_id1.get("isContainY")))

    try:
        if (node_id1.get("isContainY")):
            featureNumX = int(node_id2.get("featureNum"))
            matchColNumX = int(node_id2.get("matchColNum"))
            path_x = node_id2.get("storagePath")

            record_num = int(node_id2.get("fileRecord"))

            featureNumY = int(node_id1.get("featureNum"))
            matchColNumY = int(node_id1.get("matchColNum"))
            path_y = node_id1.get("storagePath")
        else:
            if not node_id2.get("isContainY"):
                CommonConfig.error_logger.error("both isContainY are False")
            featureNumY = int(node_id2.get("featureNum"))
            matchColNumY = int(node_id2.get("matchColNum"))
            path_y = node_id2.get("storagePath")
            record_num = int(node_id2.get("fileRecord"))

            featureNumX = int(node_id1.get("featureNum"))
            matchColNumX = int(node_id1.get("matchColNum"))
            path_x = node_id1.get("storagePath")

        CommonConfig.http_logger.info("path_x:" + str(path_x))
        CommonConfig.http_logger.info("path_y:" + str(path_y))

        path_x = os.path.join(absolute_path, path_x)
        path_y = os.path.join(absolute_path, path_y)

        train_batch_num = epoch_num * record_num // batch_size + 1
        feature_num = featureNumX + featureNumY

        CommonConfig.http_logger.info("path_x:" + str(path_x))
        CommonConfig.http_logger.info("path_y:" + str(path_y))
        CommonConfig.http_logger.info("train_batch_num:" + str(train_batch_num))
        CommonConfig.http_logger.info("feature_num:" + str(feature_num))
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
                'RS'])

        CommonConfig.http_logger.info("train_lr/run:  config:" + str(config))
        tfe.set_config(config)
        players = ['XOwner', 'YOwner', 'RS']
        prot = tfe.protocol.SecureNN(*tfe.get_config().get_players(players))
        tfe.set_protocol(prot)
        # session_target = sys.argv[2] if len(sys.argv) > 2 else None

        if (featureNumY == 0):

            x_train = prot.define_local_computation(player='XOwner', computation_fn=get_data_x,
                                                    arguments=(batch_size, path_x, featureNumX,
                                                               matchColNumX, epoch_num * 2, 3.0, 1))
            y_train = prot.define_local_computation(player='YOwner', computation_fn=get_data_y, 
                                                    arguments=(batch_size, path_y, matchColNumY,
                                                               epoch_num * 2, 1))
        else:
            x_train1, y_train = prot.define_local_computation(player='YOwner', computation_fn=get_data_xy,
                                                              arguments=(batch_size, path_y, featureNumY,
                                                                         matchColNumY, epoch_num * 2, 3.0, 1))
            x_train0 = prot.define_local_computation(player='XOwner', computation_fn=get_data_x,
                                                     arguments=(batch_size, path_x, featureNumX, matchColNumX,
                                                                epoch_num * 2, 3.0, 1))
            x_train = prot.concat([x_train0, x_train1], axis=1)

        # print("x_train:", x_train)
        # print("y_train:", y_train)
        CommonConfig.http_logger.info("x_train:" + str(x_train))
        CommonConfig.http_logger.info("y_train:" + str(y_train))

        model = LogisticRegression(feature_num, learning_rate=learningRate)

        CommonConfig.http_logger.info("modelFilePath:" + str(modelFilePath))
        CommonConfig.http_logger.info("modelFileMachine:" + str(modelFileMachine))
        CommonConfig.http_logger.info("modelFilePlainTextPath:" + str(modelFilePlainTextPath))

        save_op = model.save(modelFilePath, modelFileMachine)
        save_as_plaintext_op = model.save_as_plaintext(modelFilePlainTextPath, modelFileMachine)
        # load_op = model.load(modelFilePath, modelFileMachine)

        CommonConfig.http_logger.info("save_op:" + str(save_op))
        # with tfe.Session() as sess:
        try:
            sess = KE.get_session()
            # sess.run(tfe.global_variables_initializer(), tag='init')
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())
        except Exception as e:
            CommonConfig.error_logger.exception(
                'global_variables_initializer error, exception msg:{}'.format(str(e)))

        CommonConfig.http_logger.info("start_time:")
        start_time = time.time()
        CommonConfig.http_logger.info("start_time:" + str(start_time))

        CommonConfig.http_logger.info("train_lr/run: x_train:" + str(x_train))
        CommonConfig.http_logger.info("train_lr/run: y_train:" + str(y_train))
        CommonConfig.http_logger.info("train_lr/run: train_batch_num:" + str(train_batch_num))

        model.fit(sess, x_train, y_train, train_batch_num, progress_file)

        train_time = time.time() - start_time
        print("train_time=", train_time)

        print("Saving model...")
        sess.run(save_op)
        sess.run(save_as_plaintext_op)
        print("Save OK.")

        with open(progress_file, "w") as f:
            f.write("1.00")
            f.flush()
        # sess.close()
    except Exception as e:
        CommonConfig.error_logger.exception(
            'train.run() error, exception msg:{}'.format(str(e)))


if __name__ == '__main__':

    with open('./qqq/conf', 'r') as f:
        conf = f.read()
        print(conf)
    conf = conf.replace("True", "true").replace("False", "false")
    # print(input)
    conf = json.loads(conf)
    print(conf)

    import time

    start_time=time.time()
    # run(taskId="qqq", conf=conf, modelFileMachine="YOwner",
    #     modelFilePath="./qqq/model",
    #     modelFilePlainTextPath="./qqq/model/plaintext_model", tf_config_file="/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted/tfe_keeper/qqq/config.json")
    run(taskId="qqq", conf=conf, modelFileMachine="YOwner",
        modelFilePath="./qqq/model",
        modelFilePlainTextPath="./qqq/model/plaintext_model")

    end_time=time.time()

    print("time=", end_time-start_time)


    # run(taskId="qqq", conf=conf, modelFileMachine="YOwner",
    # modelFilePath="./qqq/model", modelFilePlainTextPath="./qqq/model/plaintext_model",
    # tf_config_file="/app/file/tfe/qqq/config.json")
