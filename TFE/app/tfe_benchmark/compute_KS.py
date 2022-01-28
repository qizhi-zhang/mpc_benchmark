# -*- coding: utf-8 -*-
from sklearn import metrics
import pandas as pd
import numpy as np

# def get_KS(label, score):
#     label=label.tolist()
#     #label=np.reshape(label, len(label))
#     score=score.tolist()
#     #score=np.reshape(score, len(label))
#     #print(label)
#     #print(score)
#     z=list(zip(label, score))
#     #print(z)
#     z.sort(key=(lambda r: r[1]), reverse=True)
#     num_T= sum(label)+1E-4
#     num_F= len(label)-num_T+1E-4
#
#     TP=0.0
#     FP=0.0
#     TPR = TP / num_T
#     FPR = FP / num_F
#     KS=0.0
#     for r in z:
#         if r[0]==1 :
#             TP=TP+1
#             TPR=TP/num_T
#         else:
#             FP=FP+1
#             FPR=FP/num_F
#
#         KS=max(KS, TPR-FPR)
#     return KS

def compute_KS_gaode3w():
    file_path='/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted/file/qqq/data'

    predict_path='/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted/'


    # y = pd.read_csv(file_path+"/公开信贷样本_目标变量含X特征变量.csv", index_col=["id"])
    # print("y:", y)
    y = pd.read_csv(file_path+"/gaode_3w_y.csv", index_col=["id","ent_date"])
    print("y:", y)

    #y_hat = pd.read_csv(predict_path+"tfe/qqq/predict", index_col=["id","ent_date"])
    y_hat = pd.read_csv(predict_path+"tfe/qqq/predict", header=None,names=["id","ent_date","predict"],index_col=["id","ent_date"])
    df = y.join(y_hat)
    #df=pd.concat([x, y], axis=1)
    print(df)


    y = df.loc[:, 'label']
    print("y=",y)
    y_hat = df.loc[:, 'predict']
    print("y_hat=", y_hat)


    y = np.array(y)
    y_hat = np.array(y_hat)




    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)


    #KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    print(KS)

def compute_KS_gaode20w():
    file_path='/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted/file/qqq/data'

    predict_path='/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted/'


    # y = pd.read_csv(file_path+"/公开信贷样本_目标变量含X特征变量.csv", index_col=["id"])
    # print("y:", y)
    y = pd.read_csv(file_path+"/gaode_20w_y.csv", index_col=["id","ent_date"])
    print("y:", y)

    #y_hat = pd.read_csv(predict_path+"tfe/qqq/predict", index_col=["id","ent_date"])
    y_hat = pd.read_csv(predict_path+"tfe/qqq/predict", header=None,names=["id","ent_date","predict"], index_col=["id","ent_date"])
    df = y.join(y_hat)
    #df=pd.concat([x, y], axis=1)
    print(df)


    y = df.loc[:, 'label']
    print("y=",y)
    y_hat = df.loc[:, 'predict']
    print("y_hat=", y_hat)


    y = np.array(y)
    y_hat = np.array(y_hat)




    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)


    #KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    print(KS)


def compute_KS_ym5w():
    file_path = '/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted/file/qqq/data'

    predict_path = '/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted/'

    y = pd.read_csv(file_path+"/embed_op_fea_5w_format_y.csv", index_col=["id","loan_date"])
    print("y:", y)
    #y = pd.read_csv(file_path + "/gaode_3w_y.csv", index_col=["id", "ent_date"])
    #print("y:", y)

    #y_hat = pd.read_csv(predict_path + "tfe/qqq/predict", index_col=["id", "ent_date"])
    y_hat = pd.read_csv(predict_path+"tfe/qqq/predict", header=None, names=["id","loan_date","predict"], index_col=["id","loan_date"])
    df = y.join(y_hat)
    # df=pd.concat([x, y], axis=1)
    print(df)

    y = df.loc[:, 'label']
    print("y=", y)
    y_hat = df.loc[:, 'predict']
    print("y_hat=", y_hat)

    y = np.array(y)
    y_hat = np.array(y_hat)

    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    print(KS)

def compute_KS_ym10w1k5():
    file_path = '/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted/file/qqq/data'

    predict_path = '/Users/qizhi.zqz/projects/TFE_zqz/tf-encrypted/'

    y = pd.read_csv(file_path+"/10w1k5col_y.csv", index_col=["oneid","loan_date"])
    print("y:", y)
    #y = pd.read_csv(file_path + "/gaode_3w_y.csv", index_col=["id", "ent_date"])
    #print("y:", y)

    #y_hat = pd.read_csv(predict_path + "tfe/qqq/predict", index_col=["id", "ent_date"])
    y_hat = pd.read_csv(predict_path+"tfe/qqq/predict", header=None, names=["oneid","loan_date","predict"], index_col=["oneid","loan_date"])
    df = y.join(y_hat)
    # df=pd.concat([x, y], axis=1)
    print(df)

    y = df.loc[:, 'label']
    print("y=", y)
    y_hat = df.loc[:, 'predict']
    print("y_hat=", y_hat)

    y = np.array(y)
    y_hat = np.array(y_hat)

    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    print(KS)


def compute_KS_xd():
    file_path = '../data/'

    predict_path = './qqq/predict'

    y = pd.read_csv(file_path+"/xindai_xy_test.csv", index_col=["id"])
    print("y:", y)
    y_hat = pd.read_csv(predict_path, header=None, names=["id","predict"], index_col=["id"])
    df = y.join(y_hat)
    # df=pd.concat([x, y], axis=1)
    print(df)

    y = df.loc[:, 'y']
    print("y=", y)
    y_hat = df.loc[:, 'predict']
    print("y_hat=", y_hat)

    y = np.array(y)
    y_hat = np.array(y_hat)

    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    # KS=get_KS(y, y_hat)
    KS = max(tpr - fpr)
    print(KS)


if __name__=='__main__':
    compute_KS_xd()