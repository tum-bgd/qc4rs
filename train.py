import os
import sys

import pandas as pd
#import shutil
import numpy as np
#import random
import time
import matplotlib.pyplot as plt
import cv2
#import multiprocessing

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

import cirq
import sympy

from skimage.transform import downscale_local_mean
from sklearn.decomposition import PCA, FactorAnalysis

from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow_quantum as tfq

# SCRIPT IMPORTS
from autoencoder import ConvAutoencoder_256, ConvAutoencoder_64, SimpleAutoencoder_256, DeepAutoencoder_64, \
    DeepAutoencoder_256, SimpleAutoencoder_64
from Circuits.embeddings import basis_embedding, angle_embedding
from Circuits.farhi import create_fvqc
from Circuits.grant import create_gvqc
from dae import DAE
from rbm import train_rbm
from data_orga import organize_data

import scipy.optimize as sopt



DEVICE = torch.device('cpu')

batch_size = 32


def train(dataset, dataset_path, classes, compression, cparam, vgg16, embedding, eparam, train_layer, loss, observable,
          optimi, grayscale, image_count, image_size, split):
    """Logging"""
    log_path = os.path.join('logs/RUN_' + str(dataset) + '_' + str(classes[0]) + 'vs' + str(classes[1]) + '_' +
                            str(compression) + '_' + 'vgg16' + str(vgg16) + '_' + str(embedding) + str(eparam) + '_' +
                            str(train_layer) + '_' + str(loss) + '_' + str(observable))
    os.mkdir(log_path)
    sys.stdout = open(log_path + '/output_log.txt', 'w')
    csv_logger = CSVLogger(log_path + '/model_log.csv', append=True, separator=';')

    start = time.time()
    print('OA timer started at:', start)

    """Preparation"""
    BATCH_SIZE = 32
    latent_dim = 16  # for autoencoder
    EPOCHS = 50  # 50 - 100 EPOCHS, MAYBE TUNE?

    organize_data(dataset_name=dataset, input_path=dataset_path, classes=classes, split=split)

    base_dir = './' + dataset + '_data_' + classes[0] + '_' + classes[1]
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'valid')

    train_count = image_count[0] + image_count[1] - 2 * split
    test_count = split
    val_count = split

    train_features, train_labels = extract_features(train_dir, train_count, image_size)
    test_features, test_labels = extract_features(test_dir, test_count, image_size)
    val_features, val_labels = extract_features(val_dir, val_count, image_size)

    # train_dl = ImageDataGenerator()
    # test_dl = ImageDataGenerator()
    # val_dl = ImageDataGenerator()

    print('Total Number of ' + str(classes[0]) + ' and ' + str(classes[1]) + ' TRAIN images is:' +
          str(len(train_features)))
    print('Total Number of ' + str(classes[0]) + ' and ' + str(classes[1]) + ' TEST images is:' +
          str(len(test_features)))
    print('Total Number of ' + str(classes[0]) + ' and ' + str(classes[1]) + ' VALIDATION images is:' +
          str(len(val_features)))

    r, c = train_labels.shape

    print('Labels are:' + str(train_labels.shape))

    if c > 2:
        train_labels = shorten_labels(train_labels)
        test_labels = shorten_labels(test_labels)
        val_labels = shorten_labels(val_labels)

    y_train = single_label(train_labels)
    y_test = single_label(test_labels)
    y_val = single_label(val_labels)

    print('Label ok?:' + str(y_train[0]) + 'and' + str(y_train[1]) + 'and' + str(y_train[2]) + 'and' + str(y_train[3]))

    if loss == 'hinge' or loss == 'squarehinge':
        # Convert labels from 1, 0 to 1, -1
        y_train = 2.0 * y_train - 1.0
        y_test = 2.0 * y_test - 1.0
        y_val = 2.0 * y_val - 1.0

    # ----------------------------------------------------------------------------------------------------------------------

    time_1 = time.time()
    passed = time_1 - start
    print('Elapsed time for preperation:', passed)

    """USE GRAYSCALE IMAGES"""
    if grayscale and compression != 'ds':
        print('Images BRG2GRAY')
        x_train = []
        x_test = []
        x_val = []

        k = 0
        for img in train_features:
            x_train.append(cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY))
            k += 1
        k = 0
        for img in test_features:
            x_test.append(cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY))
            k += 1
        k = 0
        for img in val_features:
            x_val.append(cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY))
            k += 1

        train_features = np.asarray(x_train)
        test_features = np.asarray(x_test)
        val_features = np.asarray(x_val)

    """DOWNSAMPLING"""
    if compression == 'ds':
        print('Starting dimensional reduction with downsampling!')

        # Convert to single illuminance channel
        _, train_s1, train_s2, _ = train_features.shape
        _, test_s1, test_s2, _ = test_features.shape
        _, val_s1, val_s2, _ = val_features.shape

        x_train = np.zeros((train_count, train_s1, train_s2))
        x_test = np.zeros((test_count, test_s1, test_s2))
        x_val = np.zeros((val_count, val_s1, val_s2))

        k = 0
        for img in train_features:
            x_train[k] = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)
            k += 1
        k = 0
        for img in test_features:
            x_test[k] = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)
            k += 1
        k = 0
        for img in val_features:
            x_val[k] = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)
            k += 1

        # Downsampling
        # https://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html
        ds_param = int(image_size[0] / 4)

        encoded_x_train = np.zeros((train_count, 4, 4))
        i = 0
        for img in x_train:
            encoded_x_train[i] = downscale_local_mean(img, (ds_param, ds_param))
            i += 1
        encoded_x_test = np.zeros((test_count, 4, 4))
        i = 0
        for img in x_test:
            encoded_x_test[i] = downscale_local_mean(img, (ds_param, ds_param))
            i += 1
        encoded_x_val = np.zeros((val_count, 4, 4))
        i = 0
        for img in x_val:
            encoded_x_val[i] = downscale_local_mean(img, (ds_param, ds_param))
            i += 1

    """PRINCIPAL COMPONENT ANALYSIS"""
    if compression == 'pca':
        print('Starting dimensional reduction with PCA!')

        x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count, test_count,
                                              val_count)

        pca = PCA(n_components=16)

        pca.fit(x_train)

        encoded_x_train = pca.transform(x_train)
        encoded_x_test = pca.transform(x_test)
        encoded_x_val = pca.transform(x_val)

        encoded_x_train = encoded_x_train.reshape(train_count, 4, 4)
        encoded_x_test = encoded_x_test.reshape(test_count, 4, 4)
        encoded_x_val = encoded_x_val.reshape(val_count, 4, 4)

    """AUTOENCODER"""
    if compression == 'simple_ae':
        x_train, x_test, x_val = flatten_gray_data(train_features, test_features, val_features, train_count, test_count,
                                                   val_count)

        autoencoder = Autoencoder_simple_64(latent_dim)

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

        encoded_x_train = encoded_x_train_.reshape(train_count, 4, 4)
        encoded_x_test = encoded_x_test_.reshape(test_count, 4, 4)
        encoded_x_val = encoded_x_val_.reshape(val_count, 4, 4)

    if compression == 'ae':
        if vgg16:
            print('Starting dimensional reduction with VGG16 and autoencoder!')

            if dataset == 'eurosat':
                x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count,
                                                      test_count,
                                                      val_count)
                autoencoder = Autoencoder_3_64(latent_dim)

            if dataset == 'resisc45':
                x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count,
                                                      test_count,
                                                      val_count)
                autoencoder = Autoencoder_1_256(latent_dim)

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

        if not vgg16:
            print('Starting dimensional reduction with convolutional autoencoder!')

            x_train = train_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            x_test = test_features.reshape(-1, image_size[0], image_size[1], image_size[2])
            x_val = val_features.reshape(-1, image_size[0], image_size[1], image_size[2])

            if image_size[0] == 256:
                autoencoder = ConvAutoencoder_256(latent_dim, image_size)

            if image_size[0] == 64:
                autoencoder = ConvAutoencoder_64(latent_dim, image_size)

            if image_size[0] != 256 and image_size[0] != 64:
                print('No matching autoencoder for image size' + str(image_size[0]) + 'found!')

            autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

            autoencoder.fit(x_train, x_train,
                            epochs=10,
                            shuffle=True,
                            validation_data=(x_test, x_test),
                            workers=multiprocessing.cpu_count()
                            )

            encoded_x_train_ = batch_encode_array(autoencoder, x_train, 10)  # BATCH ENCODING IF CUDA OUT OF MEMORY
            encoded_x_test_ = autoencoder.encoder(x_test).numpy()
            encoded_x_val_ = autoencoder.encoder(x_val).numpy()

        encoded_x_train = encoded_x_train_.reshape(train_count, 4, 4)
        encoded_x_test = encoded_x_test_.reshape(test_count, 4, 4)
        encoded_x_val = encoded_x_val_.reshape(val_count, 4, 4)

    """DEEP AUTOENCODER"""
    if compression == 'dae':
        print('Starting dimensional reduction with deep autoencoder!')

        seed_everything(42)

        if vgg16 and dataset == 'eurosat':
            num = 2 * 2 * 512
        if vgg16 and dataset == 'resisc45':
            num = 8 * 8 * 512
        if not vgg16:
            if not grayscale:
                num = image_size[0] * image_size[1] * image_size[2]  # Number of values: e.g. 64x64x3
            if grayscale:
                num = image_size[0] * image_size[1]  # Number of values: e.g. 64x64

        if grayscale:
            x_train, x_test, x_val = flatten_gray_data(train_features, test_features, val_features, train_count,
                                                       test_count,
                                                       val_count)
        if not grayscale:
            x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count, test_count,
                                                  val_count)
        x_train_binary, x_test_binary, x_val_binary = binarization(x_train, x_test, x_val)

        train_dl = DataLoader(
            TensorDataset(torch.Tensor(x_train_binary).to(DEVICE)),
            batch_size=32,
            shuffle=False
        )

        hidden_dimensions = [
            {
                "hidden_dim": 1000,
                "num_epochs": 10,
                "learning_rate": 0.1,
                "use_gaussian": False
            },
            {
                "hidden_dim": 500,
                "num_epochs": 10,
                "learning_rate": 0.1,
                "use_gaussian": False
            },
            {
                "hidden_dim": 250,
                "num_epochs": 10,
                "learning_rate": 0.1,
                "use_gaussian": False
            },
            {
                "hidden_dim": 16,
                "num_epochs": 30,
                "learning_rate": 0.001,  # Use much lower LR for gaussian to avoid exploding gradient
                "use_gaussian": True
                # Use a Gaussian distribution for the last hidden layer to let it take advantage of continuous values
            }
        ]

        # get initial iteration of new training dl
        new_train_dl = train_dl
        visible_dim = num
        hidden_dim = None
        models = []  # trained RBM models
        for configs in hidden_dimensions:

            # parse configs
            hidden_dim = configs["hidden_dim"]
            num_epochs = configs["num_epochs"]
            lr = configs["learning_rate"]
            use_gaussian = configs["use_gaussian"]

            # train RBM
            # print(f"{visible_dim} to {hidden_dim}")
            print(str(visible_dim) + ' to ' + str(hidden_dim))
            model, v, v_pred = train_rbm(new_train_dl, visible_dim, hidden_dim, k=1, num_epochs=num_epochs, lr=lr,
                                         use_gaussian=use_gaussian)
            models.append(model)

            # rederive new data loader based on hidden activations of trained model
            new_data = []
            for data_list in new_train_dl:
                p = model.sample_h(data_list[0])[0]
                new_data.append(p.detach().cpu().numpy())
            new_input = np.concatenate(new_data)
            new_train_dl = DataLoader(
                TensorDataset(torch.Tensor(new_input).to(DEVICE)),
                batch_size=32,
                shuffle=False
            )

            # update new visible_dim for next RBM
            visible_dim = hidden_dim

        # FINE TUNE AUTOENCODER
        lr = 1e-3
        dae = DAE(models).to(DEVICE)
        dae_loss = nn.MSELoss()
        optimizer = optim.Adam(dae.parameters(), lr)
        num_epochs = 50

        encoded = []
        # train
        for epoch in range(num_epochs):
            losses = []
            for i, data_list in enumerate(train_dl):
                data = data_list[0]
                v_pred, v_encode = dae(data)
                encoded.append(v_encode)
                batch_loss = dae_loss(data, v_pred)  # difference between actual and reconstructed
                losses.append(batch_loss.item())
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            running_loss = np.mean(losses)
            print('Epoch', epoch, ':', running_loss)

        # ENCODE DATA
        encoded_x_train = dae_encoding(x_train_binary, dae)
        encoded_x_test = dae_encoding(x_test_binary, dae)
        encoded_x_val = dae_encoding(x_val_binary, dae)

    if compression == 'fa':
        print('Starting dimensional reduction with FACTOR ANALYSIS!')

        x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count, test_count,
                                              val_count)

        fa = FactorAnalysis(n_components=16, svd_method='lapack')

        fa.fit(x_train)

        encoded_x_train = fa.transform(x_train)
        encoded_x_test = fa.transform(x_test)
        encoded_x_val = fa.transform(x_val)

        encoded_x_train = encoded_x_train.reshape(train_count, 4, 4)
        encoded_x_test = encoded_x_test.reshape(test_count, 4, 4)
        encoded_x_val = encoded_x_val.reshape(val_count, 4, 4)

    if compression == None:
        print('Please chose a dimensional reduction method! ds, pca, ae, dae')
        return

    # ----------------------------------------------------------------------------------------------------------------------

    time_2 = time.time()
    passed = time_2 - time_1
    print('Elapsed time for data compression:', passed)

    enc_x_train_u = unique2D_subarray(encoded_x_train)
    enc_x_test_u = unique2D_subarray(encoded_x_test)
    enc_x_val_u = unique2D_subarray(encoded_x_val)
    print("Encoded unique arrays: Train", enc_x_train_u.shape, "and: Test", enc_x_test_u.shape, "and: Val",
          enc_x_val_u.shape)

    """QUANTUM EMBEDDING"""
    if embedding == 'basis' or embedding == 'bin':
        x_train_bin, x_test_bin, x_val_bin = binarization(encoded_x_train, encoded_x_test, encoded_x_val)

        """CHECK HOW MANY UNIQUE ARRAYS ARE LEFT AFTER ENCODING"""
        x_train_u = unique2D_subarray(x_train_bin)
        x_test_u = unique2D_subarray(x_test_bin)
        x_val_u = unique2D_subarray(x_val_bin)
        print("Unique arrays after thresholding: Train", x_train_u.shape, "and: Test", x_test_u.shape, "and: Val",
              x_val_u.shape)

    if embedding == 'basis':
        print('Basis embedding!')
        x_train_circ = [basis_embedding(x) for x in x_train_bin]
        x_test_circ = [basis_embedding(x) for x in x_test_bin]
        x_val_circ = [basis_embedding(x) for x in x_val_bin]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        x_val_tfcirc = tfq.convert_to_tensor(x_val_circ)

    if embedding == 'angle':
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

    if embedding == 'bin':
        print('No embedding!')
        x_train_tfcirc = x_train_bin
        x_test_tfcirc = x_test_bin
        x_val_tfcirc = x_val_bin

    if embedding == 'no':
        print('No embedding!')
        x_train_tfcirc = encoded_x_train
        x_test_tfcirc = encoded_x_test
        x_val_tfcirc = encoded_x_val

    if embedding == None:
        print('Pleaes choose quantum embedding method! basis, angle, no')
        return

    # ----------------------------------------------------------------------------------------------------------------------

    time_3 = time.time()
    passed = time_3 - time_2
    print('Elapsed time for quantum embedding:', passed)

    """MODEL BUILDING"""
    if train_layer == 'farhi':
        circuit, readout = create_quantum_model(observable)

    if train_layer == 'grant':
        circuit, readout = create_quantum_model(observable)

    if train_layer == 'dense':
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(4, 4, 1)),
            tf.keras.layers.Dense(2, activation='relu'),
            tf.keras.layers.Dense(1),
        ])

    if train_layer != 'farhi' and train_layer != 'grant' and train_layer != 'dense':
        print('Chose a trainig layer! farhi, grant, dense')
        return

    if train_layer == 'farhi' or train_layer == 'grant':
        model = tf.keras.Sequential([
            # The input is the data-circuit, encoded as a tf.string
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            # The PQC layer returns the expected value of the readout gate, range [-1,1].
            tfq.layers.PQC(circuit, readout),
        ])

    if loss == 'hinge':
        print('Hinge loss selected!')
        model_loss = tf.keras.losses.Hinge()

    if loss == 'squarehinge':
        print('Square hinge loss selected!')
        model_loss = tf.keras.losses.SquaredHinge()

    if loss == 'crossentropy':
        model_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.0)
        '''
        from_logits: Whether to interpret y_pred as a tensor of logit values. By default, we assume that y_pred contains
         probabilities (i.e., values in [0, 1]). 

        label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0, we compute the loss between the 
        predicted labels and a smoothed version of the true labels, where the smoothing squeezes the labels towards 0.5.
         Larger values of label_smoothing correspond to heavier smoothing. 

        axis:  The axis along which to compute crossentropy (the features axis). Defaults to -1. 

        reduction:  Type of tf.keras.losses.Reduction to apply to loss. Default value is AUTO. AUTO indicates that the 
        reduction option will be determined by the usage context. For almost all cases this defaults to 
        SUM_OVER_BATCH_SIZE. When used with tf.distribute.Strategy, outside of built-in training loops such as tf.keras 
        compile and fit, using AUTO or SUM_OVER_BATCH_SIZE will raise an error. Please see this custom training tutorial
         for more details. 

        name:  Name for the op. Defaults to 'binary_crossentropy'. 
        '''

    if loss == None:
        print('Chose a loss function! hinge, squarehinge')
        return

    if optimi == 'adam':
        model_optimizer = tf.keras.optimizers.Adam()

    if optimi == 'bobyqa':
        model_optimizer = 0

    if optimi == None:
        print('Chose an optimizer!')
        return

    print('Compiling model .....')
    if train_layer == 'dense':
        model.compile(
            loss=model_loss,
            optimizer=model_optimizer,
            metrics=['accuracy'])
    if train_layer == 'farhi' or train_layer == 'grant':
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
    if train_layer == 'farhi' or train_layer == 'grant':
        plt.figure(figsize=(10, 5))
        plt.plot(qnn_history.history['hinge_accuracy'], label='Accuracy')
        plt.plot(qnn_history.history['val_hinge_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(log_path + '/acc.png')

    if train_layer == 'dense':
        plt.figure(figsize=(10, 5))
        plt.plot(qnn_history.history['accuracy'], label='nn accuracy')
        plt.plot(qnn_history.history['val_accuracy'], label='nn val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(log_path + '/acc.png')

    plt.figure(figsize=(10, 5))
    plt.plot(qnn_history.history['loss'], label='Loss')
    plt.plot(qnn_history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.legend()
    plt.savefig(log_path + '/loss.png')

    # model.save(log_path + '/model.h5') NOT IMPLEMENTED !!!??? https://github.com/tensorflow/quantum/issues/56
    model.save_weights(log_path + '/weights.h5')
    print('Model weights saved!')

    y_true = y_val
    y_pred = model.predict(x_val_tfcirc)

    if loss == 'hinge' or loss == 'squarehinge':
        # Hinge labels to 0,1
        y_true = (y_true + 1) / 2
        y_pred = (np.array(y_pred) + 1) / 2

        # Round Labels for Metrics
        y_pred_int = []
        for i in range(0, len(y_pred)):
            y_pred_int.append(round(y_pred[i][0]))

    if loss == 'crossentropy':
        y_true = tf.squeeze(y_true) > 0.5
        y_pred_int = tf.squeeze(y_pred) > 0.5

    precision_0 = precision_score(y_true, y_pred_int, pos_label=0, average='binary')
    recall_0 = recall_score(y_true, y_pred_int, pos_label=0, average='binary')
    f1_0 = f1_score(y_true, y_pred_int, pos_label=0, average='binary')

    precision_1 = precision_score(y_true, y_pred_int, pos_label=1, average='binary')
    recall_1 = recall_score(y_true, y_pred_int, pos_label=1, average='binary')
    f1_1 = f1_score(y_true, y_pred_int, pos_label=1, average='binary')

    print('Precision for class ', classes[0], ' is: ', precision_0)
    print('Recall for class ', classes[0], ' is: ', recall_0)
    print('F1 for class ', classes[0], ' is: ', f1_0)

    print('Precision for class ', classes[1], ' is: ', precision_1)
    print('Recall for class ', classes[1], ' is: ', recall_1)
    print('F1 for class ', classes[1], ' is: ', f1_1)


# ----------------------------------------------------------------------------------------------------------------------


def extract_features(directory, sample_count, image_size):
    if dataset == 'eurosat':
        if vgg16 and compression != 'ds':
            features = np.zeros(shape=(sample_count, 2, 2, 512))
        if not vgg16:
            features = np.zeros(shape=(sample_count, image_size[0], image_size[1], image_size[2]))
    if dataset == 'resisc45':
        if vgg16 and compression != 'ds':
            features = np.zeros(shape=(sample_count, 8, 8, 512))
        if not vgg16:
            features = np.zeros(shape=(sample_count, image_size[0], image_size[1], image_size[2]))

    labels_2 = np.zeros(shape=(sample_count, 2))  # how to find label size ?
    labels_3 = np.zeros(shape=(sample_count, 3))  # how to find label size ?

    if vgg16 and compression != 'ds':
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

    generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(directory,
                                                                         target_size=(image_size[0], image_size[1]),
                                                                         batch_size=batch_size,
                                                                         class_mode='categorical')

    i = 0

    print('Entering for loop...')

    for inputs_batch, labels_batch in generator:

        if vgg16 and compression != 'ds':
            features_batch = conv_base.predict(inputs_batch)
        if not vgg16:
            features_batch = inputs_batch
        if compression == 'ds':
            features_batch = inputs_batch

        features[i * batch_size: (i + 1) * batch_size] = features_batch
        try:
            labels_2[i * batch_size: (i + 1) * batch_size] = labels_batch
            labels = labels_2
        except:
            labels_3[i * batch_size: (i + 1) * batch_size] = labels_batch
            labels = labels_3

        i += 1
        if i * batch_size >= sample_count:
            break

    return features, labels


def shorten_labels(labels):
    labels_ = []

    for i in range(0, len(labels)):
        labels_.append(labels[i][1:])

    return labels_


def single_label(labels_):
    labels = []

    for i in range(0, len(labels_)):
        labels.append(labels_[i][0])
    labels = np.array(labels)

    return labels


def batch_encode_array(autoencoder, array, frac):  # because not enough memory to process 1100x256x256xX
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
    b = np.ascontiguousarray(a.reshape(a.shape[0], -1)).view(dtype1)

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

    x_train = np.reshape(train_features, (train_count, train_s1 * train_s2 * train_s3))
    x_test = np.reshape(test_features, (test_count, test_s1 * test_s2 * test_s3))
    x_val = np.reshape(val_features, (val_count, val_s1 * val_s2 * val_s3))

    return x_train, x_test, x_val


def flatten_gray_data(train_features, test_features, val_features, train_count, test_count, val_count):
    train_s1, train_s2, train_s3 = train_features.shape
    test_s1, test_s2, test_s3 = test_features.shape
    val_s1, val_s2, val_s3 = val_features.shape

    x_train = np.reshape(train_features, (train_s1, train_s2 * train_s3))
    x_test = np.reshape(test_features, (test_s1, test_s2 * test_s3))
    x_val = np.reshape(val_features, (val_s1, val_s2 * val_s3))

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


if __name__ == '__main__':

    train_layer = 'grant'
    vgg16 = True
    cparam = 0
    embedding = 'angle'
    eparam = 'y'
    loss = 'squarehinge'
    observable = 'z'
    optimi = 'adam'
    grayscale = False
    compression = 'fa'

    dataset = 'eurosat'
    dataset_path = '../DATASETS/EuroSAT/2750'
    image_size = [64, 64, 3]
    classes = ['AnnualCrop', 'SeaLake']
    image_count = [3000, 3000]
    split = 900

    try:
        train(dataset, dataset_path, classes, compression, cparam, vgg16, embedding, eparam, train_layer, loss,
              observable, optimi, grayscale, image_count, image_size, split)
    except FileExistsError:
        print('FILE EXISTS!')

