from wasserstein_attack import wasserstein_PGD
from utils import create_classifier, readucr, read_dataset
import os
import numpy as np
import sys
import sklearn
from sklearn import preprocessing

import tensorflow.keras as keras
import tensorflow as tf

root_dir = ".."
archive_name = "UCR"
dataset = sys.argv[1]
classifier_name = sys.argv[2]
itr = "_itr_8"
wass = sys.argv[3]
wass_clip = sys.argv[4]
norm_clip = sys.argv[5]
norm = sys.argv[6]
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

model_path = root_dir + '/results/' + classifier_name + '/' + archive_name + itr + '/' + dataset + '/'

model = keras.models.load_model(model_path+'best_model.hdf5')
was_attack = wasserstein_PGD(model)
attack_succ = 0
adv_save = []
for i in range(x_train.shape[0]):
    x = x_train[i]
    y = y_train[i]
    y_ori = y_true[i]
    x_adv = was_attack.generate(x, y, y_ori, 
                                wass = wass,  
                                norm_clips = float(norm_clip),
                                wass_clips = float(wass_clip),
                                norm = norm)
    #import pdb; pdb.set_trace()
    x_adv = tf.transpose(x_adv, [1,0])
    y_adv = model(x_adv)
    if np.argmax(y_adv.numpy())!=y_ori:
        attack_succ+=1
        adv_save.append([i, x_adv])
    
print (dataset, classifier_name, wass, x_train.shape[0], attack_succ)

np.save("./adv_result/{}_{}_{}_{}_{}.npy".format(dataset, classifier_name, wass, wass_clip, norm_clip), adv_save)

