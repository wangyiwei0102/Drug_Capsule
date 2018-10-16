#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import input_data
from dbn_tf import DBN
import numpy as np
import time
from capsNet import CapsNet
from config import cfg
import tensorflow as tf
import os
import utils
from sklearn.metrics import confusion_matrix
import math


debug = False
isTraining = True
do_grids = True
do_k_fold = False

train_datadir = "data_set"
setFileNames = ['train.csv', 'test.csv', 'test.csv']
grids = [([128, 64], (100, 100, 200), (0.001, 0.001),148, [8, 2, 2], 2)]
data = input_data.read_data_sets(train_datadir, setFileNames = setFileNames, one_hot=True)
trX, trY, teX, teY, vaX, vaY = data.train.images, data.train.labels, data.test.images, \
                               data.test.labels, data.validation.images, data.validation.labels



def eva(nn, trX, trY):
    pre = nn.predict(trX)
    pre = np.reshape(pre , (-1, 1))
    true = np.argmax(trY, axis=1)
    matrix = confusion_matrix(true, pre, labels=[1, 0])
    TP = matrix[0][0]
    TN = matrix[1][1]
    FP = matrix[1][0]
    FN = matrix[0][1]
    SE = 1.0*  TP  / (TP + FN)
    SP = 1.0 * TN / (TN + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    ACC = 1.0*(TP + TN) / (TP + TN + FP + FN)
    return TP, TN, FP, FN, SE, SP, MCC,ACC

def pre(nn, trX):
    pre = nn.predict(trX)
    pre = np.reshape(pre, (1, -1))
    # true = np.argmax(trY, axis=1)
    return pre

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

# trX, trY = shuffle_in_unison(trX, trY)


t = str(int(time.time()))


input_size = trX.shape[1]
utils.print_out('data_set : %s, len: %d'%(train_datadir, input_size))

if do_k_fold:
    k_fold = 5
    # k_fold_X = np.concatenate((trX, vaX))
    # k_fold_Y = np.concatenate((trY, vaY))
    k_fold_X = trX
    k_fold_Y = trY
    k_fold_ind = np.arange(len(k_fold_X))
    np.random.shuffle(k_fold_ind)
    k_fold_split_ind = np.split(k_fold_ind, k_fold)
    k_fold_X = [k_fold_X[i_k_fold_ind] for i_k_fold_ind in k_fold_split_ind]
    k_fold_Y = [k_fold_Y[i_k_fold_ind] for i_k_fold_ind in k_fold_split_ind]

def train(do_k_fold, out_dir, log_f):
    if do_k_fold:
        utils.print_out("# do k_fold k=%d" % k_fold, log_f)
        k_fold_val = 0
        k_fold_tra = 0
        [k_TP, k_TN, k_FP, k_FN, k_SE, k_SP, k_MCC, k_ACC] = [0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(k_fold):
            trX_=[]
            trY_=[]
            for j in range(k_fold):
                if j == i: continue
                trX_.append(k_fold_X[j])
                trY_.append(k_fold_Y[j])
            trX_ = np.concatenate(trX_)
            trY_ = np.concatenate(trY_)
            utils.print_out("#k_fold %d" % i, log_f)
            utils.print_out("#do DBN ...", log_f)
            dbn = DBN()
            dbn.train(trX_)
            utils.print_out("#end DBN", log_f)
            utils.print_out("#do caps ...", log_f)
            capsNet = CapsNet(is_training=True, dbn=dbn)

            i_k_fold_val, i_k_fold_tra = capsNet.train(trX_, trY_, k_fold_X[i], k_fold_Y[i], None, log_f)
            TP, TN, FP, FN, SE, SP, MCC, ACC = eva(capsNet, k_fold_X[i], k_fold_Y[i])
            print(i,", TP:", TP)
            print(i,", TN:", TN)
            print(i,", FP:", FP)
            print(i,", FN:", FN)
            print(i,", SE:", SE)
            print(i,", SP:", SP)
            print(i,", MCC:", MCC)
            print(i,", ACC: ", ACC)
            k_TP += TP
            k_TN += TN
            k_FP += FP
            k_FN += FN
            k_SE += SE
            k_SP += SP
            k_MCC += MCC
            k_ACC += ACC

        print("TP :", k_TP / 5)
        print("TN :", k_TN / 5)
        print("FP :", k_FP / 5)
        print("FN :", k_FN / 5)
        print("SE :", k_SE / 5)
        print("SP :", k_SP / 5)
        print("MCC: ", k_MCC / 5)
        print("ACC: ", k_ACC / 5)
    else:
        utils.print_out("#do DBN ...", log_f)
        dbn = DBN()
        dbn.train(trX)
        utils.print_out("#end DBN", log_f)
        utils.print_out("#do caps ...", log_f)
        utils.print_out("#test instead val set for test ...", log_f)
        capsNet = CapsNet(is_training=isTraining, dbn=dbn)
        if isTraining:
            i_k_fold_val, i_k_fold_tra = capsNet.train(trX, trY, teX, teY, "./board", log_f)
            utils.print_out("#end caps", log_f)

            tr_TP, tr_TN, tr_FP, tr_FN, tr_SE, tr_SP, tr_MCC, tr_ACC = eva(capsNet, trX, trY)
            val_TP, val_TN, val_FP, val_FN, val_SE, val_SP, val_MCC, val_ACC = eva(capsNet, vaX, vaY)
            te_P, te_TN, te_FP, te_FN, te_SE, te_SP, te_MCC,te_ACC = eva(capsNet, teX, teY)
            utils.print_out('train : TP:%.3f;   TN:%.3f;      FP:%.3f;     FN:%.3f;  SE:%.3f  SP:%.3f   MCC:%.3f  P:%.3f' \
                            %(tr_TP, tr_TN, tr_FP, tr_FN, tr_SE, tr_SP, tr_MCC, tr_ACC), log_f)
            utils.print_out('val : TP:%.3f;   TN:%.3f;      FP:%.3f;      FN:%.3f;  SE:%.3f  SP:%.3f   MCC:%.3f P:%.3f' \
                            % (val_TP, val_TN, val_FP, val_FN, val_SE, val_SP, val_MCC, val_ACC), log_f)
            utils.print_out('test : TP:%.3f;   TN:%.3f;      FP:%.3f;      FN:%.3f;  SE:%.3f  SP:%.3f   MCC:%.3f P:%.3f' \
                            % (te_P, te_TN, te_FP, te_FN, te_SE, te_SP, te_MCC, te_ACC), log_f)

        else:
            import csv
            csvFile = open("./"+train_datadir+"/"+setFileNames[1], "r")
            reader = csv.reader(csvFile)  # 返回的是迭代类型
            data = []
            for item in reader:
                data.append(item[0])
            csvFile.close()
            data = data[1:]

            utils.print_out("#end caps", log_f)
            pre_Y= pre(capsNet, vaX).tolist()[0]
            import pandas as pd

            dataFrame = pd.DataFrame({ "0_name": data,"1_class": pre_Y})
            dataFrame.to_csv('./data_set/test_dir/180831-result.csv', index=False, sep=",")

if do_grids:
    for i , i_grid in enumerate(grids):
        utils.print_out('For grid %d param : %s' % (i, str(i_grid)))
        nn_szie, nums, lr, batch_size, caps,  iter_routing = i_grid

        cfg.input_size=input_size
        cfg.batch_size=batch_size
        cfg.nn_hsizes=nn_szie
        cfg.nn_layer_size=len(nn_szie)
        cfg.dbn_learning_rate=lr[0]
        cfg.caps_startLr=lr[1]
        cfg.dbn_epoches=nums[0]
        cfg.caps_epochs=nums[1]
        cfg.caps_decay_steps=nums[2]
        cfg.primaryCaps_out_num=caps[0]
        cfg.primaryCaps_vec_num=caps[1]
        cfg.outCaps_vec_num=caps[2]
        cfg.iter_routing=iter_routing

        if not debug:
            t = str(int(time.time()))
            out_dir = os.path.join(cfg.out_dir, "log_%s" % t)

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            log_file = os.path.join(out_dir, "log_%s" % t)
            log_f = tf.gfile.GFile(log_file, mode="a")
            utils.print_out("# log_file=%s" % log_file, log_f)
        else:
            log_f = None
            out_dir = None

        utils.print_hparams(cfg, f=log_f)
        train(do_k_fold, out_dir, log_f)
        utils.print_out('END grid %d' % i)

else:
    if not debug:
        t = str(int(time.time()))
        out_dir = os.path.join(cfg.out_dir, "log_%s" % t)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        log_file = os.path.join(out_dir, "log_%s" % t)
        log_f = tf.gfile.GFile(log_file, mode="a")
        utils.print_out("# log_file=%s" % log_file, log_f)
    else:
        log_f = None
        out_dir = None
    utils.print_hparams(cfg, f=log_f)
    train(do_k_fold, out_dir, log_f)




