import os
import sys

import pandas as pd
import shutil
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import cv2
import multiprocessing

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from tensorflow.keras.callbacks import CSVLogger

import tensorflow_quantum as tfq

import cirq
import sympy

from skimage.transform import downscale_local_mean
from sklearn.decomposition import PCA, FactorAnalysis

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

#from sklearn.multiclass import OneVsRestClassifier

# SCRIPT IMPORTS
from Preprocessing.autoencoderModels import ConvAutoencoder_256, ConvAutoencoder_64, SimpleAutoencoder_64, SimpleAutoencoder_256, DeepAutoencoder_64, DeepAutoencoder_256
from Circuits.embeddings import basis_embedding, angle_embedding
from Circuits.farhi import create_fvqc
from Circuits.grant import create_gvqc
from Preprocessing.dae import DAE
from Preprocessing.rbm import train_rbm
from organizeData import organize_data_ovr

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
BATCH_SIZE = 32

def dim_reduc(dataset, train_layer, train_features, test_features, val_features, base_dir):  
    
    print('Starting dimensional reduction with VGG16 and autoencoder!')

    latent_dim = 16
    
    if dataset == 'eurosat':
        if train_layer == 'farhi':
            
            x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count, test_count,
                                                  val_count)
            autoencoder = DeepAutoencoder_64(latent_dim)

        if train_layer == 'grant':
            x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count, test_count,
                                              val_count)
            autoencoder = DeepAutoencoder_64(latent_dim)
            
        if train_layer == 'dense':
            x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count, test_count,
                                                  val_count)
            autoencoder = DeepAutoencoder_64(latent_dim)
    
        autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
        autoencoder.fit(x_train, x_train,
                        epochs=50,
                        shuffle=True,
                        validation_data=(x_test, x_test),
                        workers=multiprocessing.cpu_count()
                        )
        
        encoded_x_train_ = autoencoder.encoder(x_train).numpy()
        encoded_x_test_ = autoencoder.encoder(x_test).numpy()
        encoded_x_val_ = autoencoder.encoder(x_val).numpy()
            
        
    if dataset == 'resisc45':
        if train_layer == 'farhi':
            #print('Starting dimensional reduction with factor analysis!')

            #train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count, test_count, val_count)
            x_train = train_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            x_test = test_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            x_val = val_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            
            train_dir = os.path.join(base_dir, 'train')
            test_dir = os.path.join(base_dir, 'test')
            val_dir = os.path.join(base_dir, 'valid')
            
            train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(train_dir,
                                                                 target_size=(image_size[0], image_size[1]),
                                                                 batch_size=32,
                                                                 classes = classes,
                                                                 class_mode='input')
            
            test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(test_dir,
                                                                 target_size=(image_size[0], image_size[1]),
                                                                 batch_size=32,
                                                                 classes = classes,
                                                                 class_mode='input')
            '''
            val_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(val_dir,
                                                                 target_size=(image_size[0], image_size[1]),
                                                                 batch_size=2,
                                                                 classes = classes,
                                                                 class_mode='input')
            '''
            autoencoder = ConvAutoencoder_256(latent_dim, image_size)
            
            autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
            autoencoder.fit(train_generator,
                            epochs=50,
                            shuffle=True,
                            validation_data=test_generator,
                            workers=multiprocessing.cpu_count()
                            )

            encoded_x_train_ = batch_encode_array(autoencoder, x_train, 441)  # MAGIC NUMBER: 22050/50
            encoded_x_test_ = batch_encode_array(autoencoder, x_test, 189)  # MAGIC NUMBER: 4725/25
            encoded_x_val_ = batch_encode_array(autoencoder, x_val, 189)
            
            '''
            fa = FactorAnalysis(n_components=16)#, svd_method='lapack')

            fa.fit(train_generator)

            encoded_x_train_ = fa.transform(train_generator)
            encoded_x_test_ = fa.transform(test_generator)
            encoded_x_val_ = fa.transform(val_generator)
            '''
        if train_layer == 'grant':
            print('Starting dimensional reduction with convolutional autoencoder!')

            x_train = train_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            x_test = test_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            x_val = val_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            
            autoencoder = ConvAutoencoder_256(latent_dim, image_size)

            autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

            train_dir = os.path.join(base_dir, 'train')
            test_dir = os.path.join(base_dir, 'test')
            val_dir = os.path.join(base_dir, 'valid')
            
            train_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(train_dir,
                                                                 target_size=(image_size[0], image_size[1]),
                                                                 batch_size=64,
                                                                 classes = classes,
                                                                 class_mode='input')
    
            test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(test_dir,
                                                                 target_size=(image_size[0], image_size[1]),
                                                                 batch_size=64,
                                                                 classes = classes,
                                                                 class_mode='input')
        
            autoencoder.fit(train_generator,
                            epochs=10,
                            shuffle=True,
                            validation_data=test_generator,
                            workers=multiprocessing.cpu_count()
                            )

            encoded_x_train_ = batch_encode_array(autoencoder, x_train, 441)  # MAGIC NUMBER: 22050/50
            encoded_x_test_ = batch_encode_array(autoencoder, x_test, 189)  # MAGIC NUMBER: 4725/25
            encoded_x_val_ = batch_encode_array(autoencoder, x_val, 189)

    encoded_x_train = encoded_x_train_.reshape(int(train_count), 4, 4)
    encoded_x_test = encoded_x_test_.reshape(int(test_count), 4, 4)
    encoded_x_val = encoded_x_val_.reshape(int(val_count), 4, 4)

    enc_x_train_u = unique2D_subarray(encoded_x_train)
    enc_x_test_u = unique2D_subarray(encoded_x_test)
    enc_x_val_u = unique2D_subarray(encoded_x_val)
    print("Encoded unique arrays: Train", enc_x_train_u.shape, "and: Test", enc_x_test_u.shape, "and: Val",
          enc_x_val_u.shape)

    return encoded_x_train, encoded_x_test, encoded_x_val
    
    
def quantum_embedding(train_layer, encoded_x_train, encoded_x_test, encoded_x_val):
    """QUANTUM EMBEDDING"""
    if train_layer == 'dense':
        #x_train_bin, x_test_bin, x_val_bin = binarization(encoded_x_train, encoded_x_test, encoded_x_val)

        """CHECK HOW MANY UNIQUE ARRAYS ARE LEFT AFTER ENCODING"""
        #x_train_u = unique2D_subarray(x_train_bin)
        #x_test_u = unique2D_subarray(x_test_bin)
        #x_val_u = unique2D_subarray(x_val_bin)
        #print("Unique arrays after thresholding: Train", x_train_u.shape, "and: Test", x_test_u.shape, "and: Val", 
        #      x_val_u.shape)
        #print('No embedding!')
        #x_train_tfcirc = x_train_bin
        #x_test_tfcirc = x_test_bin
        #x_val_tfcirc = x_val_bin
        x_train_tfcirc = encoded_x_train
        x_test_tfcirc = encoded_x_test
        x_val_tfcirc = encoded_x_val
        
    if train_layer == 'farhi':
        '''
        x_train_bin, x_test_bin, x_val_bin = binarization(encoded_x_train, encoded_x_test, encoded_x_val)

        """CHECK HOW MANY UNIQUE ARRAYS ARE LEFT AFTER ENCODING"""
        x_train_u = unique2D_subarray(x_train_bin)
        x_test_u = unique2D_subarray(x_test_bin)
        x_val_u = unique2D_subarray(x_val_bin)
        print("Unique arrays after thresholding: Train", x_train_u.shape, "and: Test", x_test_u.shape, "and: Val", 
              x_val_u.shape)
        
        print('Basis embedding!')
        x_train_circ = [basis_embedding(x) for x in np.asarray(x_train_bin)]
        x_test_circ = [basis_embedding(x) for x in np.asarray(x_test_bin)]
        x_val_circ = [basis_embedding(x) for x in np.asarray(x_val_bin)]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        x_val_tfcirc = tfq.convert_to_tensor(x_val_circ)
        '''
        eparam = 'y'
        print(eparam, 'Angle embedding!')
        train_maximum = np.max(np.abs(encoded_x_train))
        test_maximum = np.max(np.abs(encoded_x_test))
        val_maximum = np.max(np.abs(encoded_x_val))
        x_train_norm = encoded_x_train / train_maximum
        x_test_norm = encoded_x_test / test_maximum
        x_val_norm = encoded_x_val / val_maximum

        x_train_circ = [angle_embedding(x, eparam) for x in x_train_norm]
        x_test_circ = [angle_embedding(x, eparam) for x in x_test_norm]
        x_val_circ = [angle_embedding(x, eparam) for x in x_val_norm]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        x_val_tfcirc = tfq.convert_to_tensor(x_val_circ)
        
    if train_layer == 'grant':
        eparam = 'x'
        print(eparam, 'Angle embedding!')
        train_maximum = np.max(np.abs(encoded_x_train))
        test_maximum = np.max(np.abs(encoded_x_test))
        val_maximum = np.max(np.abs(encoded_x_val))
        x_train_norm = encoded_x_train / train_maximum
        x_test_norm = encoded_x_test / test_maximum
        x_val_norm = encoded_x_val / val_maximum

        x_train_circ = [angle_embedding(x, eparam) for x in x_train_norm]
        x_test_circ = [angle_embedding(x, eparam) for x in x_test_norm]
        x_val_circ = [angle_embedding(x, eparam) for x in x_val_norm]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        x_val_tfcirc = tfq.convert_to_tensor(x_val_circ)
        
    return x_train_tfcirc, x_test_tfcirc, x_val_tfcirc
    
    
def train(dataset, one_class, rest_classes, x_train_tfcirc, x_test_tfcirc, x_val_tfcirc, y_train, y_test, y_val, train_layer):  
    """LOGGING"""
    csv_logger = CSVLogger(log_path + '/model_log_' + str(one_class) + '.csv', append=True, separator=';')

    """PREPARATION"""
    EPOCHS = 3
    
    if train_layer == 'farhi' or train_layer == 'grant':
        y_train = 2 * y_train - 1
        y_test = 2 * y_test - 1
        y_val = 2 * y_val - 1

# ----------------------------------------------------------------------------------------------------------------------

    """MODEL BUILDING"""
    if train_layer == 'farhi':
        observable = 'x'
        circuit, readout = create_fvqc(observable)

    if train_layer == 'grant':
        observable = 'x'
        circuit, readout = create_gvqc(observable)

    if train_layer == 'dense':
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(4, 4, 1)),
            tf.keras.layers.Dense(2, activation='relu'),
            tf.keras.layers.Dense(1),
        ])

    if train_layer == 'farhi' or train_layer == 'grant':
        model = tf.keras.Sequential([
            # The input is the data-circuit, encoded as a tf.string
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            # The PQC layer returns the expected value of the readout gate, range [-1,1].
            tfq.layers.PQC(circuit, readout),
        ])

    if train_layer == 'farhi':
        print('Hinge loss selected!')
        model_loss = tf.keras.losses.SquaredHinge()

    if train_layer == 'grant':
        print('Square hinge loss selected!')
        model_loss = tf.keras.losses.SquaredHinge()

    if train_layer == 'dense':
        print('Binary cross entropy loss selected!')
        model_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.0)

    model_optimizer = tf.keras.optimizers.Adam()
    
    print('Compiling model .....')
    model.compile(
        loss=model_loss,
        optimizer=model_optimizer,
        metrics=[hinge_accuracy])

    qnn_history = model.fit(
        x_train_tfcirc, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test_tfcirc, y_test),
        callbacks=[csv_logger])

    time_4 = time.time()
    passed = time_4 - time_3
    print('Elapsed time for training:', passed)
    passed = time_4 - start
    print('OA elapsed time:', passed)

    print('Model training completed!')

    qnn_results = model.evaluate(x_val_tfcirc, y_val)
    print(qnn_results)
    print('Model evaluated!')

    
    # SAVE PLOTS FOR ACC AND LOSS
    plt.figure(figsize=(10,5))
    plt.plot(qnn_history.history['hinge_accuracy'], label='qnn accuracy')
    plt.plot(qnn_history.history['val_hinge_accuracy'], label='qnn val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(log_path + '/acc_' + str(one_class) + '.png')
    
    plt.figure(figsize=(10,5))
    plt.plot(qnn_history.history['loss'], label='qnn loss')
    plt.plot(qnn_history.history['val_loss'], label='qnn val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('val_loss')
    plt.legend()
    plt.savefig(log_path + '/loss_' + str(one_class) + '.png')

    # model.save(log_path + '/model.h5') NOT IMPLEMENTED !!!??? https://github.com/tensorflow/quantum/issues/56
    model.save_weights(log_path + '/weights_' + str(one_class) + '.h5')
    print('Model weights saved!')
    
    y_true_ = y_val
    y_pred = model.predict(x_val_tfcirc)
    
    print(y_true_)
    print(y_pred)
    
    if train_layer == 'farhi' or train_layer == 'grant':
        # Hinge labels to 0,1
        y_true_ = (y_true_ + 1)/2
        y_pred = (np.array(y_pred) + 1)/2
 
        y_true = []
        for i in range(len(y_true_)):
            y_true.append(int(y_true_[i]))
        y_true = np.asarray(y_true)

        # Round Labels for Metrics
        y_pred_int = []
        for i in range(0, len(y_pred)):    
            y_pred_int.append(int(round(y_pred[i][0])))

        precision_0 = precision_score(y_true, y_pred_int, pos_label=0, average='binary')
        recall_0 = recall_score(y_true, y_pred_int, pos_label=0, average='binary')
        f1_0 = f1_score(y_true, y_pred_int, pos_label=0, average='binary')

        precision_1 = precision_score(y_true, y_pred_int, pos_label=1, average='binary')
        recall_1 = recall_score(y_true, y_pred_int, pos_label=1, average='binary')
        f1_1 = f1_score(y_true, y_pred_int, pos_label=1, average='binary')

        print('Precision for class ', one_class ,' is: ', precision_0)
        print('Recall for class ', one_class ,' is: ', recall_0)
        print('F1 for class ', one_class ,' is: ', f1_0)

        print('Precision for class for 0 labels is: ', precision_1)
        print('Recall for 0 labels is: ', recall_1)
        print('F1 for 0 labels is: ', f1_1)

        tmp = set(y_true)-set(y_pred_int)
        print('Value from set(y_true)-set(y_pred_int): SHOWS VALUES FROM PRED THAT ARE NOT IN TRUTH:', tmp)
    
    print('-----------------------Training of model for ', str(one_class) , ' finished!-----------------------')
    
    return model
    
# ----------------------------------------------------------------------------------------------------------------------


def extract_features(dataset, directory, sample_count, image_size, classes, vgg16):
    if dataset == 'eurosat':
        if vgg16:
            features = np.zeros(shape=(sample_count, 2, 2, 512))
        if not vgg16:
            features = np.zeros(shape=(sample_count, image_size[0], image_size[1], image_size[2]))
    if dataset == 'resisc45':
        if vgg16:
            features = np.zeros(shape=(sample_count, 8, 8, 512))
        if not vgg16:
            features = np.zeros(shape=(sample_count, image_size[0], image_size[1], image_size[2]))

    labels = np.zeros(shape=(sample_count, len(classes)))
    
    if vgg16:
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

    generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(directory,
                                                                         target_size=(image_size[0], image_size[1]),
                                                                         batch_size=batch_size,
                                                                         classes = classes,
                                                                         class_mode='categorical')

    i = 0

    print('Entering for loop...')

    for inputs_batch, labels_batch in generator:

        if vgg16:
            features_batch = conv_base.predict(inputs_batch)
        if not vgg16:
            features_batch = inputs_batch
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
       
        i += 1
        if i * batch_size >= sample_count:
            break

    return features, labels, generator.class_indices


def batch_encode_array(autoencoder, array, frac): # because not enough memory to process 1100x256x256xX
    cut = int(len(array) / frac)
    encoded = []
    j = 0

    for i in range(1, frac + 1):
        tmp = autoencoder.encoder(array[j * cut:i * cut]).numpy()
        encoded.append(tmp)
        j = i

    encoded_array = np.asarray(encoded)

    return encoded_array


def unique2D_subarray(a):
    dtype1 = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    b = np.ascontiguousarray(a.reshape(a.shape[0],-1)).view(dtype1)

    return a[np.unique(b, return_index=1)[1]]


def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


def flatten_data(train_features, test_features, val_features, train_count, test_count, val_count):
    _, train_s1, train_s2, train_s3 = train_features.shape
    _, test_s1, test_s2, test_s3 = test_features.shape
    _, val_s1, val_s2, val_s3 = val_features.shape

    x_train = np.reshape(train_features, (int(train_count), train_s1 * train_s2 * train_s3))
    x_test = np.reshape(test_features, (int(test_count), test_s1 * test_s2 * test_s3))
    x_val = np.reshape(val_features, (int(val_count), val_s1 * val_s2 * val_s3))

    return x_train, x_test, x_val


def seed_everything(seed=42):
    """Seed everything to make the code more reproducable.

    This code is the same as that found from many public Kaggle kernels.

    Parameters
    ----------
    seed: int
        seed value to ues

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def binarization(encoded_x_train, encoded_x_test, encoded_x_val):
    print('Binarization if inputs...')
    unique_tmp = np.unique(encoded_x_train)
    THRESHOLD = np.median(unique_tmp)

    print("Threshold for Binarization is:", THRESHOLD)
    x_train_bin = np.array(encoded_x_train > THRESHOLD, dtype=np.float32)
    x_test_bin = np.array(encoded_x_test > THRESHOLD, dtype=np.float32)
    x_val_bin = np.array(encoded_x_val > THRESHOLD, dtype=np.float32)

    return x_train_bin, x_test_bin, x_val_bin


def dae_encoding(x_binary, dae):
    encoded_x = []
    len_input = len(x_binary)
    input_test = torch.Tensor(x_binary[0:len_input]).to(DEVICE)

    transformed, tmp = dae.encode(input_test)

    for i in range(0, len(transformed)):
        encoded_x.append(transformed[i].detach().cpu().numpy())
    encoded_x = np.array(encoded_x).reshape(len_input, 4, 4)

    return encoded_x


def train_ovr(args):
    '''
    dataset = 'resisc45'

    if dataset == 'resisc45':
        dataset_path = '../DATASETS/RESISC45/RESISC45_data'  
        train_layer = 'farhi'
        #classes = ['beach', 'dense_residential', 'thermal_power_station', 'snowberg', 'meadow', 'desert', 'island', 'industrial_area', 'forest', 'rectangular_farmland', 'freeway', 'river', 'lake', 'circular_farmland', 'sparse_residential']
        classes = ['storage_tank', 'beach', 'palace', 'airport', 'dense_residential', 'tennis_court', 'thermal_power_station', 'ship', 'chaparral', 'bridge', 'snowberg', 'roundabout', 'commercial_area', 'sea_ice', 'meadow', 'intersection', 'basketball_court', 'golf_course', 'ground_track_field', 'desert', 'railway_station', 'mobile_home_park', 'parking_lot', 'island', 'airplane', 'harbor', 'cloud', 'mountain', 'industrial_area', 'forest', 'rectangular_farmland', 'medium_residential', 'church', 'overpass', 'freeway', 'baseball_diamond', 'river', 'wetland', 'railway', 'runway', 'lake', 'stadium', 'circular_farmland', 'terrace', 'sparse_residential']
        #classes = ['dense_residential', 'thermal_power_station', 'meadow', 'beach', 'lake']
        image_count = np.ones(45)*700
        image_size = [256, 256, 3]
        split = 0.3
        vgg16 = False
    '''
    
    dataset = 'eurosat'

    if dataset == 'eurosat':
        dataset_path = '../DATASETS/EuroSAT/2750'  
        train_layer = 'farhi'
        
        classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
        image_count = [3000, 3000, 3000, 2500, 2500, 2000, 2500, 3000, 2500, 3000]
        #classes = ['SeaLake', 'Industrial', 'Pasture']
        #image_count = [3000, 2500, 2000]
        image_size = [64, 64, 3]
        
        split = 0.3
        vgg16 = True
    
    
    #--------------------------------------------------------------------------------------------------------
    
    """LOGGING"""
    log_path = os.path.join('logs/RUN_OneVsRest_' + str(dataset) + '_' + str(train_layer))
    os.mkdir(log_path)
    sys.stdout = open(log_path + '/output_log.txt', 'w')
    
    start = time.time()
    print('OA timer started at:', start)
    
    #--------------------------------------------------------------------------------------------------------
    # DATA PREPARATION
    
    organize_data_ovr(dataset_name=dataset, input_path=dataset_path, classes=classes, split=split)
        
    base_dir = './' + dataset + '_data_OvR'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'valid')

    train_count = 0
    test_count = 0
    val_count = 0
    for i in range(len(classes)):
        train_count += image_count[i] - image_count[i] * split
        test_count += (image_count[i] * split)/2
        val_count += (image_count[i] * split)/2

    train_features, train_labels_, train_class_indices = extract_features(dataset, train_dir, int(train_count), image_size, classes, vgg16)
    test_features, test_labels_, test_class_indices = extract_features(dataset, test_dir, int(test_count), image_size, classes, vgg16)
    val_features, val_labels_, val_class_indices = extract_features(dataset, val_dir, int(val_count), image_size, classes, vgg16)

    print('Total Number of TRAIN images is:' + str(len(train_features)))
    print('Total Number of TEST images is:' + str(len(test_features)))
    print('Total Number of VALIDATION images is:' + str(len(val_features)))
    
    train_labels = []
    for i in range(len(train_labels_)):
        train_labels.append(np.argmax(train_labels_[i]))
        
    test_labels = []
    for i in range(len(test_labels_)):
        test_labels.append(np.argmax(test_labels_[i]))

    val_labels = []
    for i in range(len(val_labels_)):
        val_labels.append(np.argmax(val_labels_[i]))
            
    time_1 = time.time()
    passed = time_1 - start
    print('Elapsed time for preperation:', passed)
    
    #--------------------------------------------------------------------------------------------------------
    # DATA PREPROCESSING
    
    x_train, x_test, x_val = dim_reduc(dataset, train_layer, train_features, test_features, val_features, base_dir)
        
    time_2 = time.time()
    passed = time_2 - time_1
    print('Elapsed time for dimensionality reduction:', passed)

    #--------------------------------------------------------------------------------------------------------
    
    x_train_tfcirc, x_test_tfcirc, x_val_tfcirc = quantum_embedding(train_layer, x_train, x_test, x_val)
    
    time_3 = time.time()
    passed = time_3 - time_2
    print('Elapsed time for quantum embedding:', passed)
    
    #--------------------------------------------------------------------------------------------------------
    # MODEL TRAINING
    
    models = []
    for one_class in classes:
        rest_classes = classes[:]
        rest_classes.remove(one_class)
        
        one_class_int = train_class_indices[one_class]

        # CHANGE LABELS TO 1 FOR ONE CLASS AND 0 FOR EVERY OTHER CLASS
        y_train = []
        y_test = []
        y_val = []
        for i in range(len(train_labels)):
            if train_labels[i] == one_class_int:
                y_train.append(1)
            if train_labels[i] != one_class_int:
                y_train.append(0)

        for i in range(len(test_labels)):
            if test_labels[i] == one_class_int:
                y_test.append(1)
            if test_labels[i] != one_class_int:
                y_test.append(0)
                
        for i in range(len(val_labels)):
            if val_labels[i] == one_class_int:
                y_val.append(1)
            if val_labels[i] != one_class_int:
                y_val.append(0)
        
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        y_val = np.asarray(y_val)

        model = train(dataset, one_class, rest_classes, x_train_tfcirc, x_test_tfcirc, x_val_tfcirc, y_train, y_test, y_val, train_layer)
        models.append(model)
        
        time_temp = time.time()
        print('A model finished training at:', time_temp)
    
    time_4 = time.time()
    print('All models finished training at:', time_4)
    passed = time_4 - time_3
    print('Elapsed time for training all models:', passed)

    #--------------------------------------------------------------------------------------------------------
    # EVALUATION
    preds = []
    for model in models:
        pred = model.predict(x_val_tfcirc)
        preds.append(pred)  # [model1_pred, model2_pred, ...]
        
    preds_sorted = []
    for i in range(len(x_val_tfcirc)):
        models_together = []
        for pred in preds:
            models_together.append(pred[i])
        preds_sorted.append(models_together)
        
    ovr_pred_values = []
    ovr_preds = []
    for pred in preds_sorted:  
        ovr_pred_values.append(max(pred)) 
        ovr_preds.append(np.argmax(pred)) # The model with the highest value for the predicted class
    
    
    set_test = set(val_labels)-set(ovr_preds)
    print('set(val_labels)-set(ovr_preds)', set_test)
    
    # CONFUSION MATRIX
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    #labelsToDisplay = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    cm = confusion_matrix(val_labels, ovr_preds)
    
    '''
    labelsarray-like of shape (n_classes), default=None

    List of labels to index the matrix. This may be used to reorder or select a subset of labels. 
    If None is given, those that appear at least once in y_true or y_pred are used in sorted order.
    
    sample_weightarray-like of shape (n_samples,), default=None

    Sample weights.
    New in version 0.18.
    
    normalize{‘true’, ‘pred’, ‘all’}, default=None

    Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. 
    If None, confusion matrix will not be normalized.
    '''
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.savefig(log_path + '/confusion_matrix_' + str(train_layer) + str(dataset) + '.png')
    print('Confusion Matrix')
    print(classes)
    
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    print('Overall:[ FP, FN, TP, TN ] [', FP, FN, TP, TN, ']')

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print('Sensitivity, hit rate, recall, or true positive rate: ', TPR)

    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    print('Specificity or true negative rate: ', TNR)

    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print('Precision or positive predictive value: ', PPV)

    # Negative predictive value
    NPV = TN/(TN+FN)
    print('Negative predictive value: ', NPV)

    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print('Fall out or false positive rate: ', FPR)

    # False negative rate
    FNR = FN/(TP+FN)
    print('False negative rate: ', FNR)

    # False discovery rate
    FDR = FP/(TP+FP)
    print('False discovery rate: ', FDR)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print('Overall accuracy: ', ACC)

if __name__ == '__main__':
    '''
    dataset = 'resisc45'

    if dataset == 'resisc45':
        dataset_path = '../DATASETS/RESISC45/RESISC45_data'  
        train_layer = 'farhi'
        #classes = ['beach', 'dense_residential', 'thermal_power_station', 'snowberg', 'meadow', 'desert', 'island', 'industrial_area', 'forest', 'rectangular_farmland', 'freeway', 'river', 'lake', 'circular_farmland', 'sparse_residential']
        classes = ['storage_tank', 'beach', 'palace', 'airport', 'dense_residential', 'tennis_court', 'thermal_power_station', 'ship', 'chaparral', 'bridge', 'snowberg', 'roundabout', 'commercial_area', 'sea_ice', 'meadow', 'intersection', 'basketball_court', 'golf_course', 'ground_track_field', 'desert', 'railway_station', 'mobile_home_park', 'parking_lot', 'island', 'airplane', 'harbor', 'cloud', 'mountain', 'industrial_area', 'forest', 'rectangular_farmland', 'medium_residential', 'church', 'overpass', 'freeway', 'baseball_diamond', 'river', 'wetland', 'railway', 'runway', 'lake', 'stadium', 'circular_farmland', 'terrace', 'sparse_residential']
        #classes = ['dense_residential', 'thermal_power_station', 'meadow', 'beach', 'lake']
        image_count = np.ones(45)*700
        image_size = [256, 256, 3]
        split = 0.3
        vgg16 = False
    '''
    
    dataset = 'eurosat'

    if dataset == 'eurosat':
        dataset_path = '../DATASETS/EuroSAT/2750'  
        train_layer = 'farhi'
        
        classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
        image_count = [3000, 3000, 3000, 2500, 2500, 2000, 2500, 3000, 2500, 3000]
        #classes = ['SeaLake', 'Industrial', 'Pasture']
        #image_count = [3000, 2500, 2000]
        image_size = [64, 64, 3]
        
        split = 0.3
        vgg16 = True
    
    
    #--------------------------------------------------------------------------------------------------------
    
    """LOGGING"""
    log_path = os.path.join('logs/RUN_OneVsRest_' + str(dataset) + '_' + str(train_layer))
    os.mkdir(log_path)
    sys.stdout = open(log_path + '/output_log.txt', 'w')
    
    start = time.time()
    print('OA timer started at:', start)
    
    #--------------------------------------------------------------------------------------------------------
    # DATA PREPARATION
    
    organize_data_ovr(dataset_name=dataset, input_path=dataset_path, classes=classes, split=split)
        
    base_dir = './' + dataset + '_data_OvR'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'valid')

    train_count = 0
    test_count = 0
    val_count = 0
    for i in range(len(classes)):
        train_count += image_count[i] - image_count[i] * split
        test_count += (image_count[i] * split)/2
        val_count += (image_count[i] * split)/2

    train_features, train_labels_, train_class_indices = extract_features(dataset, train_dir, int(train_count), image_size, classes, vgg16)
    test_features, test_labels_, test_class_indices = extract_features(dataset, test_dir, int(test_count), image_size, classes, vgg16)
    val_features, val_labels_, val_class_indices = extract_features(dataset, val_dir, int(val_count), image_size, classes, vgg16)

    print('Total Number of TRAIN images is:' + str(len(train_features)))
    print('Total Number of TEST images is:' + str(len(test_features)))
    print('Total Number of VALIDATION images is:' + str(len(val_features)))
    
    train_labels = []
    for i in range(len(train_labels_)):
        train_labels.append(np.argmax(train_labels_[i]))
        
    test_labels = []
    for i in range(len(test_labels_)):
        test_labels.append(np.argmax(test_labels_[i]))

    val_labels = []
    for i in range(len(val_labels_)):
        val_labels.append(np.argmax(val_labels_[i]))
            
    time_1 = time.time()
    passed = time_1 - start
    print('Elapsed time for preperation:', passed)
    
    #--------------------------------------------------------------------------------------------------------
    # DATA PREPROCESSING
    
    x_train, x_test, x_val = dim_reduc(dataset, train_layer, train_features, test_features, val_features, base_dir)
        
    time_2 = time.time()
    passed = time_2 - time_1
    print('Elapsed time for dimensionality reduction:', passed)

    #--------------------------------------------------------------------------------------------------------
    
    x_train_tfcirc, x_test_tfcirc, x_val_tfcirc = quantum_embedding(train_layer, x_train, x_test, x_val)
    
    time_3 = time.time()
    passed = time_3 - time_2
    print('Elapsed time for quantum embedding:', passed)
    
    #--------------------------------------------------------------------------------------------------------
    # MODEL TRAINING
    
    models = []
    for one_class in classes:
        rest_classes = classes[:]
        rest_classes.remove(one_class)
        
        one_class_int = train_class_indices[one_class]

        # CHANGE LABELS TO 1 FOR ONE CLASS AND 0 FOR EVERY OTHER CLASS
        y_train = []
        y_test = []
        y_val = []
        for i in range(len(train_labels)):
            if train_labels[i] == one_class_int:
                y_train.append(1)
            if train_labels[i] != one_class_int:
                y_train.append(0)

        for i in range(len(test_labels)):
            if test_labels[i] == one_class_int:
                y_test.append(1)
            if test_labels[i] != one_class_int:
                y_test.append(0)
                
        for i in range(len(val_labels)):
            if val_labels[i] == one_class_int:
                y_val.append(1)
            if val_labels[i] != one_class_int:
                y_val.append(0)
        
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        y_val = np.asarray(y_val)

        model = train(dataset, one_class, rest_classes, x_train_tfcirc, x_test_tfcirc, x_val_tfcirc, y_train, y_test, y_val, train_layer)
        models.append(model)
        
        time_temp = time.time()
        print('A model finished training at:', time_temp)
    
    time_4 = time.time()
    print('All models finished training at:', time_4)
    passed = time_4 - time_3
    print('Elapsed time for training all models:', passed)

    #--------------------------------------------------------------------------------------------------------
    # EVALUATION
    preds = []
    for model in models:
        pred = model.predict(x_val_tfcirc)
        preds.append(pred)  # [model1_pred, model2_pred, ...]
        
    preds_sorted = []
    for i in range(len(x_val_tfcirc)):
        models_together = []
        for pred in preds:
            models_together.append(pred[i])
        preds_sorted.append(models_together)
        
    ovr_pred_values = []
    ovr_preds = []
    for pred in preds_sorted:  
        ovr_pred_values.append(max(pred)) 
        ovr_preds.append(np.argmax(pred)) # The model with the highest value for the predicted class
    
    
    set_test = set(val_labels)-set(ovr_preds)
    print('set(val_labels)-set(ovr_preds)', set_test)
    
    # CONFUSION MATRIX
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    #labelsToDisplay = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
    cm = confusion_matrix(val_labels, ovr_preds)
    
    '''
    labelsarray-like of shape (n_classes), default=None

    List of labels to index the matrix. This may be used to reorder or select a subset of labels. 
    If None is given, those that appear at least once in y_true or y_pred are used in sorted order.
    
    sample_weightarray-like of shape (n_samples,), default=None

    Sample weights.
    New in version 0.18.
    
    normalize{‘true’, ‘pred’, ‘all’}, default=None

    Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. 
    If None, confusion matrix will not be normalized.
    '''
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.savefig(log_path + '/confusion_matrix_' + str(train_layer) + str(dataset) + '.png')
    print('Confusion Matrix')
    print(classes)
    
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    print('Overall:[ FP, FN, TP, TN ] [', FP, FN, TP, TN, ']')

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print('Sensitivity, hit rate, recall, or true positive rate: ', TPR)

    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    print('Specificity or true negative rate: ', TNR)

    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    print('Precision or positive predictive value: ', PPV)

    # Negative predictive value
    NPV = TN/(TN+FN)
    print('Negative predictive value: ', NPV)

    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    print('Fall out or false positive rate: ', FPR)

    # False negative rate
    FNR = FN/(TP+FN)
    print('False negative rate: ', FNR)

    # False discovery rate
    FDR = FP/(TP+FP)
    print('False discovery rate: ', FDR)

    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print('Overall accuracy: ', ACC)