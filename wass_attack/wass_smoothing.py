from wasserstein_attack import wasserstein_PGD
from utils import create_classifier, readucr, read_dataset
import os
import numpy as np
import sys
import sklearn
from sklearn import preprocessing

import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_probability as tfp

def wass_smooth(x, model, std, N):
    m = tfp.distributions.Laplace(0, std)
    batch_size = x.shape[0]
    ##normalization in the feature dim to 0-1##
    #x = (x-tf.reduce_min(x))/(tf.reduce_max(x)-tf.reduce_min(x))
    x_expanded = tf.tile(x,[N,1])
    flow_vert = m.sample([x_expanded.shape[0], x_expanded.shape[1]-1])
    flow_vert = tf.cast(flow_vert, tf.float64) ## assign values later requires float64
    tmp = tf.Variable(x_expanded)## tensor cannot directly assign value, convert to variable
    for i in range(x_expanded.shape[1]):
        if i!=0: ##boundary
            tmp[:, i].assign(flow_vert[:,i-1]+tmp[:, i])
        if i!=x_expanded.shape[1]-1:  ##boundary
            tmp[:, i].assign(-flow_vert[:,i]+tmp[:, i])
    x_expanded_new = tf.convert_to_tensor(tmp)
    pred = model(x_expanded_new)

    return pred


def wass_smooth_certify(x, y, model, std, radi, N ):
    pred_soft = wass_smooth(x, model,std, N)
    pred_soft_mean = tf.reduce_mean(pred_soft,0)
    pred_target = pred_soft_mean[np.argmax(y)]
    pred_other = tf.math.exp(2**1.5*radi/std) * (1-pred_target)
    if pred_target>=pred_other:
        cert = 1
    else:
        cert = 0
    
    if np.argmax(y) == np.argmax(pred_soft_mean):
        conv = 1
    else:
        conv = 0
    return cert, conv


def main():
    root_dir = ".."
    archive_name = "UCR"
    dataset = sys.argv[1]
    classifier_name = sys.argv[2]
    itr = "_itr_8"
    wass_clip = sys.argv[3]
    norm_clip = sys.argv[4]
    radi = sys.argv[5]
    datasets_dict = read_dataset(root_dir, archive_name, dataset)
    x_train = datasets_dict[dataset][0]
    y_train = datasets_dict[dataset][1]
    x_test = datasets_dict[dataset][2]
    y_test = datasets_dict[dataset][3]
    if x_train.shape[0]>100:
        x_train = x_train[:100]
        y_train = y_train[:100]
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))
    # transform the labels from integers to one hot vectors
    enc = preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    input_shape = x_train.shape[1:]
    ####### prepare x_adv
    adv_path = './adv_result/' + dataset + '_' + classifier_name + '_' + 'True' + '_' + wass_clip + '_' + norm_clip + '.npy'
    x_adv = np.load(adv_path, allow_pickle = True)

    ####### prepare model
    model_path =  root_dir + '/results/' + classifier_name + '/' + archive_name + itr + '/' + dataset + '/'
    model = keras.models.load_model(model_path+'best_model.hdf5')
    ### certified accuracy on adv
    cert_all = 0
    conv_all = 0
    for i in range(x_adv.shape[0]):
        x = x_adv[i][1]
        y = y_train[x_adv[i][0]]
        cert, conv = wass_smooth_certify(x, y, model, 0.5, float(radi), 100)
        cert_all+=cert
        conv_all+=conv
    
    print(cert_all, conv_all, x_adv.shape[0])
    """
    ###certified and conventional accuracy on clean
    cert_all = 0
    conv_all = 0
    for i in range(x_train.shape[0]):
        x = x_train[i]
        y = y_train[i]
        x = tf.transpose(x, [1,0])
        cert, conv = wass_smooth_certify(x, y, model, 0.01, float(radi), 100)
        cert_all+=cert
        conv_all+=conv
    
    print(cert_all, conv_all, x_train.shape[0])
    """ 

if  __name__ == '__main__':
    main()

