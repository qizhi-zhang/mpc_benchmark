# -*- coding: utf-8 -*-
from sklearn import metrics
import pandas as pd
import numpy as np

file_path = '../data/'

predict_path = './qqq/predict'

def compute_KS_gaode3w():

    y = pd.read_csv(file_path+"/gaode_3w_y.csv", index_col=["id","ent_date"])
    print("y:", y)
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
    KS = max(tpr - fpr)
    print(KS)

def compute_KS_gaode20w():
    y = pd.read_csv(file_path+"/gaode_20w_y.csv", index_col=["id","ent_date"])
    print("y:", y)
    y_hat = pd.read_csv(predict_path+"tfe/qqq/predict", header=None,names=["id","ent_date","predict"], index_col=["id","ent_date"])
    df = y.join(y_hat)
    print(df)
    y = df.loc[:, 'label']
    print("y=",y)
    y_hat = df.loc[:, 'predict']
    print("y_hat=", y_hat)
    y = np.array(y)
    y_hat = np.array(y_hat)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)
    KS = max(tpr - fpr)
    print(KS)


def compute_KS_ym5w():
    y = pd.read_csv(file_path+"/embed_op_fea_5w_format_y.csv", index_col=["id","loan_date"])
    print("y:", y)
    y_hat = pd.read_csv(predict_path+"tfe/qqq/predict", header=None, names=["id","loan_date","predict"], index_col=["id","loan_date"])
    df = y.join(y_hat)
    print(df)
    y = df.loc[:, 'label']
    print("y=", y)
    y_hat = df.loc[:, 'predict']
    print("y_hat=", y_hat)

    y = np.array(y)
    y_hat = np.array(y_hat)

    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)

    KS = max(tpr - fpr)
    print(KS)

def compute_KS_ym10w1k5():
    y = pd.read_csv(file_path+"/10w1k5col_y.csv", index_col=["oneid","loan_date"])
    print("y:", y)
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
    KS = max(tpr - fpr)
    print(KS)


def compute_KS_xd():
    y = pd.read_csv(file_path+"/xindai_xy_test.csv", index_col=["id"])
    print("y:", y)
    y_hat = pd.read_csv(predict_path, header=None, names=["id","predict"], index_col=["id"])
    df = y.join(y_hat)
    print(df)
    y = df.loc[:, 'y']
    print("y=", y)
    y_hat = df.loc[:, 'predict']
    print("y_hat=", y_hat)
    y = np.array(y)
    y_hat = np.array(y_hat)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_hat)
    KS = max(tpr - fpr)
    print(KS)


if __name__=='__main__':
    compute_KS_xd()