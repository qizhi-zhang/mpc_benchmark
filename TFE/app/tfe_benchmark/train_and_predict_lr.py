"""Private training on combined data from several data owners"""
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import tf_encrypted as tfe
import json
# from common_private import  ModelOwner, LogisticRegression, XOwner, YOwner
from common_private import LogisticRegression
from read_data_tf import get_data_xy, get_data_x, get_data_y, get_data_id_with_y, get_data_id_with_xy
# from .common_private import LogisticRegression
# from .read_data_tf import get_data_xy, get_data_x, get_data_y, get_data_id_with_y, get_data_id_with_xy


from tf_encrypted.keras import backend as KE
import tensorflow as tf
import math
# import sys
import time
import platform
import os
#from commonutils.common_config import CommonConfig
# from ..commonutils.common_config import CommonConfig

if platform.system() == "Darwin":
    absolute_path = "/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted"
    absolute_path = "/Users/qizhi.zqz/projects/morse-tfe/morse-tfe"
else:
    absolute_path = "/app"
    # absolute_path = "/app/file"


def run(taskId, conf, modelFileMachine, modelFilePath, modelFilePlainTextPath, tf_config_file=None):

    trian_progress_file = os.path.join(absolute_path, "tfe_benchmark/" + taskId + "/train_progress")
    predict_progress_file = os.path.join(absolute_path, "tfe_benchmark/" + taskId + "/predict_progress")

    with open(trian_progress_file, "w") as f:
        f.write(str(0.0) + "\n")
        f.flush()
    with open(predict_progress_file, "w") as f:
        f.write(str(0.0) + "\n")
        f.flush()
    if not os.path.exists(trian_progress_file):
        print(
            'trian_progress_file {} does not exists'.format(trian_progress_file))

    if not os.path.exists(predict_progress_file):
        print(
            'predict_progress_file {} does not exists'.format(predict_progress_file))

    print("init train_progress_file:" + str(trian_progress_file))
    print("init  predict_progress_file:" + str(predict_progress_file))

    train_predict_Params = conf.get("trainParams")

    print("train_predict_lr/run:  train_predict_Params:" + str(train_predict_Params))

    learningRate = float(train_predict_Params.get("learningRate"))
    batch_size = int(train_predict_Params.get("batchSize"))
    epoch_num = int(train_predict_Params.get("maxIter"))
    # epsilon = float(train_predict_Params.get("epsilon"))
    # regularizationL1=float(train_predict_Params.get("regularizationL1"))
    # regularizationL2=float(train_predict_Params.get("regularizationL2"))

    dataSet = conf.get("dataSet")

    print("dataSet:" + str(dataSet))

    try:
        node_list = list(dataSet.keys())
        node_key_id1 = node_list.pop()
        node_key_id2 = node_list.pop()

        node_id1 = dataSet.get(node_key_id1)
        node_id2 = dataSet.get(node_key_id2)
    except Exception as e:
        print(
            'get node  from dataSet {} error , exception msg:{}'.format(str(dataSet), str(e)))

    # node_id1=dataSet.get("node_id1")
    # node_id2=dataSet.get("node_id2")

    # print("node1_containY:", node_id1.get("isContainY"))
    print("node1_containY:" + str(node_id1.get("isContainY")))

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
                print("both isContainY are False")
            featureNumY = int(node_id2.get("featureNum"))
            matchColNumY = int(node_id2.get("matchColNum"))
            path_y = node_id2.get("storagePath")
            record_num = int(node_id2.get("fileRecord"))

            featureNumX = int(node_id1.get("featureNum"))
            matchColNumX = int(node_id1.get("matchColNum"))
            path_x = node_id1.get("storagePath")

        print("path_x:" + str(path_x))
        print("path_y:" + str(path_y))

        path_x = os.path.join(absolute_path, path_x)
        path_y = os.path.join(absolute_path, path_y)

        train_batch_num = epoch_num * record_num // batch_size + 1
        feature_num = featureNumX + featureNumY

        print("path_x:" + str(path_x))
        print("path_y:" + str(path_y))
        print("train_batch_num:" + str(train_batch_num))
        print("feature_num:" + str(feature_num))

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

        print("train_lr/run:  config:" + str(config))
        tfe.set_config(config)
        players = ['XOwner', 'YOwner', 'RS']
        prot = tfe.protocol.SecureNN(*tfe.get_config().get_players(players))
        tfe.set_protocol(prot)

        if (featureNumY == 0):

            x_train = prot.define_local_computation(player='XOwner', computation_fn=get_data_x, 
                                                    arguments=(batch_size, path_x, featureNumX, 
                                                               matchColNumX, epoch_num * 2, 5.0, 1))
            y_train = prot.define_local_computation(player='YOwner', computation_fn=get_data_y, 
                                                    arguments=(batch_size, path_y, 
                                                               matchColNumY, epoch_num * 2, 1))

        else:
            x_train1, y_train = prot.define_local_computation(player='YOwner', computation_fn=get_data_xy, 
                                                              arguments=(batch_size, path_y, featureNumY, 
                                                                         matchColNumY, epoch_num * 2, 5.0, 1))
            x_train0 = prot.define_local_computation(player='XOwner', computation_fn=get_data_x, 
                                                     arguments=(batch_size, path_x, featureNumX, 
                                                                matchColNumX, epoch_num * 2, 5.0, 1))
            x_train = prot.concat([x_train0, x_train1], axis=1)

        # print("x_train:", x_train)
        # print("y_train:", y_train)
        print("x_train:" + str(x_train))
        print("y_train:" + str(y_train))

        model = LogisticRegression(feature_num, learning_rate=learningRate)

        print("modelFilePath:" + str(modelFilePath))
        print("modelFileMachine:" + str(modelFileMachine))
        print("modelFilePlainTextPath:" + str(modelFilePlainTextPath))

        save_op = model.save(modelFilePath, modelFileMachine)
        save_as_plaintext_op = model.save_as_plaintext(modelFilePlainTextPath, modelFileMachine)
        # load_op = model.load(modelFilePath, modelFileMachine)

        print("save_op:" + str(save_op))
        # with tfe.Session() as sess:

        # ------------------------ predict:------------------------------------------------------------

        dataSet = conf.get("dataSetPredict")
        node_list = list(dataSet.keys())
        node_key_id1 = node_list.pop()
        node_key_id2 = node_list.pop()

        node_id1 = dataSet.get(node_key_id1)
        node_id2 = dataSet.get(node_key_id2)

        print("node1_containY:", node_id1.get("isContainY"))

        if (node_id1.get("isContainY")):
            # featureNumX = int(node_id2.get("featureNum"))
            # matchColNumX = int(node_id2.get("matchColNum"))
            path_x = node_id2.get("storagePath")
            record_num = int(node_id2.get("fileRecord"))
            # 
            # featureNumY = int(node_id1.get("featureNum"))
            # matchColNumY = int(node_id1.get("matchColNum"))
            path_y = node_id1.get("storagePath")
        else:
            assert node_id2.get("isContainY")
            # featureNumY = int(node_id2.get("featureNum"))
            # matchColNumY = int(node_id2.get("matchColNum"))
            path_y = node_id2.get("storagePath")
            record_num = int(node_id2.get("fileRecord"))

            # featureNumX = int(node_id1.get("featureNum"))
            # matchColNumX = int(node_id1.get("matchColNum"))
            path_x = node_id1.get("storagePath")

        path_x = os.path.join(absolute_path, path_x)
        path_y = os.path.join(absolute_path, path_y)
        batch_num = int(math.ceil(1.0 * record_num / batch_size))
        feature_num = featureNumX + featureNumY

        print("path_x_predict:" + str(path_x))
        print("path_y_predict:" + str(path_y))

        @tfe.local_computation("XOwner")
        def provide_test_data_x(path):
            x = get_data_x(batch_size, path, featureNum=featureNumX, matchColNum=matchColNumX, epoch=2,
                           clip_by_value=5.0, skip_row_num=1)
            return x

        # @tfe.local_computation("YOwner")
        def provide_test_data_y(path):
            idx, y = get_data_id_with_y(batch_size, path, matchColNum=matchColNumX, epoch=2, skip_row_num=1)
            return idx, y

        # @tfe.local_computation("YOwner")
        def provide_test_data_xy(path):
            idx, x, y = get_data_id_with_xy(batch_size, path, featureNum=featureNumY, matchColNum=matchColNumX, epoch=2,
                                            clip_by_value=5.0, skip_row_num=1)
            return idx, x, y

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
        print("x_test:" + str(x_test))
        print("y_test:" + str(y_test))

        try:
            sess = KE.get_session()
            # sess.run(tfe.global_variables_initializer(), tag='init')
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())
        except Exception as e:
            print(
                'global_variables_initializer error , exception msg:{}'.format(str(e)))

        print("start_time:")
        start_time = time.time()
        print("start_time:" + str(start_time))

        print("train_and_predict_lr/run:  x_train:" + str(x_train))
        print("train_and_predict_lr/run:  y_train:" + str(y_train))
        print("train_and_predict_lr/run:  train_batch_num:" + str(train_batch_num))

        model.fit(sess, x_train, y_train, train_batch_num, trian_progress_file)

        train_time = time.time() - start_time
        print("train_time=", train_time)
        print("train_time=" + str(train_time))

        print("Saving model...")
        print("Saving model...")
        sess.run(save_op)
        sess.run(save_as_plaintext_op)
        print("Save OK.")
        print("Save OK.")

        with open(trian_progress_file, "a") as f:
            f.write("1.00")
            f.flush()
        # ---------------------predict:--------------------------------------

        start_time = time.time()
        print("predict start_time:" + str(start_time))

        record_num_ceil_mod_batch_size = record_num % batch_size
        if record_num_ceil_mod_batch_size == 0:
            record_num_ceil_mod_batch_size = batch_size
        model.predict(sess, x_test, os.path.join(absolute_path, "tfe_benchmark/{task_id}/predict".format(task_id=taskId)),
                      batch_num, idx, predict_progress_file, YOwner.device_name, record_num_ceil_mod_batch_size)

        test_time = time.time() - start_time
        print("predict_time=", test_time)

        print("predict_time=:" + str(test_time))

        with open(predict_progress_file, "a") as f:
            f.write("1.00")
            f.flush()

        sess.close()
    except Exception as e:
        print(
            'train_and_predict.run() error , exception msg:{}'.format(str(e)))


if __name__ == '__main__':

    # with open('./qqq/conf_xd', 'r') as f:
    #     conf = f.read()
    #     print(conf)
    conf = """
    {
    "trainParams": {
        "learningRate": 0.01,
        "maxIter": 10,
        "batchSize": 128,
        "epsilon": 0.0001,
        "regularizationL1": 0,
        "regularizationL2": 1.0,
        "solver": "tfe"
                },
    "algorithm": "LR",
    "trainSetRate": 0.8,
    "positiveValue": 1,
    "dataSet": {
        "node_id1": {
            "storagePath": "data/xindai_xx_train.csv",
            "fileRecord": 8429,
            "isContainY": False,
            "matchColNum": 1,
            "featureNum": 5
        },
        "node_id2": {
            "storagePath": "data/xindai_xy_train.csv",
            "fileRecord": 8429,
            "isContainY": True,
            "matchColNum": 1,
            "featureNum": 5
        }
    },
    "dataSetPredict": {
        "node_id1": {
            "storagePath": "data/xindai_xx_test.csv",
            "fileRecord": 3613,
            "isContainY": False,
            "matchColNum": 1,
            "featureNum": 5
        },
        "node_id2": {
            "storagePath": "data/xindai_xy_test.csv",
            "fileRecord": 3613,
            "isContainY": True,
            "matchColNum": 1,
            "featureNum": 5
        }
    }
}
    """
    conf = conf.replace("True", "true").replace("False", "false")
    # print(input)
    conf = json.loads(conf)
    print(conf)

    #run(taskId="qqq", conf=conf, modelFileMachine="YOwner",
    #    modelFilePath="./qqq/model", modelFilePlainTextPath="./qqq/model/plaintext_model")

    # three host
    run(taskId="qqq", conf=conf, modelFileMachine="YOwner",
       modelFilePath="./qqq/model", modelFilePlainTextPath="./qqq/model/plaintext_model", tf_config_file="../config.json")
