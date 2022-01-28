"""Private training on combined data from several data owners"""
import tf_encrypted as tfe
# import tensorflow as tf
import json
# from common_private import  ModelOwner, LogisticRegression, XOwner, YOwner
from common_private import LogisticRegression
from read_data_tf import get_data_xy, get_data_x, get_data_y
# from sklearn.utils import shuffle
# from sklearn.preprocessing import OneHotEncoder
# import argparse
import sys
import time


def run(taskId, algorithm, conf, modelFileMachine, modelFilePath):
    trainParams = conf.get("trainParams")

    learningRate = float(trainParams.get("learningRate"))
    batch_size = int(trainParams.get("batchSize"))
    epoch_num = int(trainParams.get("maxIter"))
    # epsilon = float(trainParams.get("epsilon"))
    # regularizationL1 = float(trainParams.get("regularizationL1"))
    # regularizationL2 = float(trainParams.get("regularizationL2"))

    dataSet = conf.get("dataSet")
    node_id1 = dataSet.get("node_id1")
    node_id2 = dataSet.get("node_id2")

    print("node1_containY:", node_id1.get("isContainY"))

    if (node_id1.get("isContainY")):
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

    batch_num = record_num // batch_size
    feature_num = featureNumX + featureNumY

    if len(sys.argv) >= 2:
        # config file was specified
        config_file = sys.argv[1]
        config = tfe.RemoteConfig.load(config_file)
    else:
        # default to using local config
        config = tfe.LocalConfig([
            'XOwner',
            'YOwner',
            'RS'])
    tfe.set_config(config)
    players = ['XOwner', 'YOwner', 'RS']
    prot = tfe.protocol.SecureNN(*tfe.get_config().get_players(players))
    tfe.set_protocol(prot)
    # session_target = sys.argv[2] if len(sys.argv) > 2 else None

    # tfe.set_protocol(tfe.protocol.Pond(
    # tfe.get_config().get_player(data_owner_0.player_name), 
    # tfe.get_config().get_player(data_owner_1.player_name)
    # ))

    @tfe.local_computation("XOwner")
    def provide_test_data_x(path="/Users/qizhi.zqz/projects/TFE/tf-encrypted/"
                            + "examples/test_on_morse_datas/data/embed_op_fea_5w_format_x.csv"):
        train_x = get_data_x(64, path, featureNum=featureNumX,
                             matchColNum=matchColNumX, epoch=epoch_num, clip_by_value=3.0, skip_row_num=1)
        return train_x

    @tfe.local_computation("YOwner")
    def provide_test_data_y(path="/Users/qizhi.zqz/projects/TFE/tf-encrypted/"
                            + "examples/test_on_morse_datas/data/embed_op_fea_5w_format_y.csv"):
        train_y = get_data_y(64, path, matchColNum=matchColNumX, epoch=epoch_num, skip_row_num=1)
        return train_y

    @tfe.local_computation("YOwner")
    def provide_test_data_xy(path="/Users/qizhi.zqz/projects/TFE/tf-encrypted/"
                             + "examples/test_on_morse_datas/data/embed_op_fea_5w_format_y.csv"):
        train_x, train_y = get_data_xy(64, path, featureNum=featureNumY,
                                       matchColNum=matchColNumX, epoch=epoch_num, clip_by_value=3.0, skip_row_num=1)
        return train_x, train_y

    if (featureNumY == 0):
        x_test = provide_test_data_x(path_x)
        y_test = provide_test_data_y(path_y)
    else:
        x_test1, y_test = provide_test_data_xy(path_y)
        x_test0 = provide_test_data_x(path_x)
        x_test = prot.concat([x_test0, x_test1], axis=1)

    print("x_train:", x_test)
    print("y_train:", y_test)
    model = LogisticRegression(feature_num, learning_rate=learningRate)

    load_op = model.load(modelFilePath, modelFileMachine)

    with tfe.Session() as sess:

        sess.run(tfe.global_variables_initializer(), 
                 tag='init')
        start_time = time.time()

        print("Loading model...")
        sess.run(load_op)
        print("Load OK.")
        # model.fit(sess, x_train, y_train, train_batch_num)
        model.get_KS(sess, x_test, y_test, batch_num)

        test_time = time.time() - start_time
        print("test_time = ", test_time)


if __name__ == '__main__':

    with open('./qqq/conf', 'r') as f:
        conf = f.read()
        print(conf)
    conf = conf.replace("True", "true").replace("False", "false")
    # print(input)
    conf = json.loads(conf)
    print(conf)

    run(taskId="qqq", algorithm="tfe_lr", conf=conf, modelFileMachine="YOwner", modelFilePath="./qqq/model")
