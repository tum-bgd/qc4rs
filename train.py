import os
import sys
from xmlrpc.client import boolean

import pandas as pd
#import shutil
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

import cirq
import sympy

from skimage.transform import downscale_local_mean
from sklearn.decomposition import PCA, FactorAnalysis

from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow_quantum as tfq

# SCRIPT IMPORTS
from Preprocessing.autoencoderModels import ConvAutoencoder_256, ConvAutoencoder_64, SimpleAutoencoder_256, DeepAutoencoder_64, \
    DeepAutoencoder_256, SimpleAutoencoder_64
from Circuits.embeddings import basis_embedding, angle_embedding
from Circuits.farhi import create_fvqc
from Circuits.grant import create_gvqc
from Preprocessing.dae import DAE
from Preprocessing.rbm import train_rbm
from organizeData import organize_data

import scipy.optimize as sopt

import argparse


def train(args):
    latent_dim = 16 # equals number of data qubits

    if args.dataset == 'eurosat':
        image_size = [64, 64, 3]

    if args.dataset == 'resisc45':
        image_size = [256, 256, 3]

    log_path = os.path.join('../logs/RUN_' + str(args.dataset) + '_' + str(args.class1) + 'vs' + str(args.class2) + '_' +
                            str(args.preprocessing) + '_' + 'vgg16' + str(args.vgg16) + '_' + str(args.embedding) + str(args.embeddingparam) + '_' +
                            str(args.train_layer) + '_' + str(args.loss) + '_' + str(args.observable))
    os.mkdir(log_path)
    sys.stdout = open(log_path + '/output_log.txt', 'w')
    csv_logger = CSVLogger(log_path + '/model_log.csv', append=True, separator=';')

    start = time.time()
    print('OA timer started at:', start)

    print('lel')
    organize_data(dataset_name=args.dataset, input_path=args.dataset_path, classes=[args.class1, args.class2], split=int(0.3*args.image_count))

    base_dir = './' + '../' + args.dataset + '_data_' + args.class1 + '_' + args.class2
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'valid')

    train_count = args.image_count + args.image_count - int(0.6*args.image_count)
    test_count = int(0.3*args.image_count)
    val_count = test_count

    train_features, train_labels = extract_features(train_dir, train_count, image_size)
    test_features, test_labels = extract_features(test_dir, test_count, image_size)
    val_features, val_labels = extract_features(val_dir, val_count, image_size)

    print('Total Number of ' + str(args.class1) + ' and ' + str(args.class2) + ' TRAIN images is:' +
          str(len(train_features)))
    print('Total Number of ' + str(args.class1) + ' and ' + str(args.class2) + ' TEST images is:' +
          str(len(test_features)))
    print('Total Number of ' + str(args.class1) + ' and ' + str(args.class2) + ' VALIDATION images is:' +
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

    if args.loss == 'hinge' or args.loss == 'squarehinge':
        # convert labels from 1, 0 to 1, -1
        y_train = 2.0 * y_train - 1.0
        y_test = 2.0 * y_test - 1.0
        y_val = 2.0 * y_val - 1.0

    # ----------------------------------------------------------------------------------------------------------------------

    time_1 = time.time()
    passed = time_1 - start
    print('Elapsed time for preperation:', passed)

    """GRAYSCALE"""
    if args.grayscale and args.preprocessing != 'ds':
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
    if args.preprocessing == 'ds':
        print('Starting dimensional reduction with downsampling!')

        # convert to single illuminance channel
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
    if args.preprocessing == 'pca':
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

    """AUTOENCODER""" # TO DO: ADD AE FOR VGG16 FALSE?????
    if args.preprocessing == 'ae':
        x_train, x_test, x_val = flatten_gray_data(train_features, test_features, val_features, train_count, test_count,
                                                   val_count)

        autoencoder = SimpleAutoencoder_64(latent_dim)

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

    if args.preprocessing == 'dae':
        if args.vgg16:
            print('Starting dimensional reduction with VGG16 and autoencoder!')

            if args.dataset == 'eurosat':
                x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count,
                                                      test_count,
                                                      val_count)
                autoencoder = DeepAutoencoder_64(latent_dim)

            if args.dataset == 'resisc45':
                x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count,
                                                      test_count,
                                                      val_count)
                autoencoder = SimpleAutoencoder_256(latent_dim)

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

        if not args.vgg16:
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
                            batch_size=args.batchsize1,
                            epochs=10,
                            shuffle=True,
                            validation_data=(x_test, x_test),
                            workers=multiprocessing.cpu_count()
                            )

            encoded_x_train_ = batch_encode_array(autoencoder, x_train, 10)  # BATCH ENCODING SINCE CUDA OUT OF MEMORY
            encoded_x_test_ = autoencoder.encoder(x_test).numpy()
            encoded_x_val_ = autoencoder.encoder(x_val).numpy()

        encoded_x_train = encoded_x_train_.reshape(train_count, 4, 4)
        encoded_x_test = encoded_x_test_.reshape(test_count, 4, 4)
        encoded_x_val = encoded_x_val_.reshape(val_count, 4, 4)

    """RBM AUTOENCODER"""
    if args.preprocessing == 'rbmae':
        print('Starting dimensional reduction with deep autoencoder!')

        seed_everything(42)

        if args.vgg16 and args.dataset == 'eurosat':
            num = 2 * 2 * 512
        if args.vgg16 and args.dataset == 'resisc45':
            num = 8 * 8 * 512
        if not args.vgg16:
            if not args.grayscale:
                num = image_size[0] * image_size[1] * image_size[2]  # Number of values: e.g. 64x64x3
            if args.grayscale:
                num = image_size[0] * image_size[1]  # Number of values: e.g. 64x64

        if args.grayscale:
            x_train, x_test, x_val = flatten_gray_data(train_features, test_features, val_features, train_count,
                                                       test_count,
                                                       val_count)
        if not args.grayscale:
            x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count, test_count,
                                                  val_count)
        x_train_binary, x_test_binary, x_val_binary = binarization(x_train, x_test, x_val)

        train_dl = DataLoader(
            TensorDataset(torch.Tensor(x_train_binary).to(args.device)),
            batch_size=args.batchsize1,
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
                TensorDataset(torch.Tensor(new_input).to(args.device)),
                batch_size=args.batchsize1,
                shuffle=False
            )

            # update new visible_dim for next RBM
            visible_dim = hidden_dim

        # FINE TUNE AUTOENCODER
        lr = 1e-3
        dae = DAE(models).to(args.device)
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

    if args.preprocessing == 'fa':
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

    if args.preprocessing == None:
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
    if args.embedding == 'basis' or args.embedding == 'bin':
        x_train_bin, x_test_bin, x_val_bin = binarization(encoded_x_train, encoded_x_test, encoded_x_val)

        """CHECK HOW MANY UNIQUE ARRAYS ARE LEFT AFTER ENCODING"""
        x_train_u = unique2D_subarray(x_train_bin)
        x_test_u = unique2D_subarray(x_test_bin)
        x_val_u = unique2D_subarray(x_val_bin)
        print("Unique arrays after thresholding: Train", x_train_u.shape, "and: Test", x_test_u.shape, "and: Val",
              x_val_u.shape)

    if args.embedding == 'basis':
        print('Basis embedding!')
        x_train_circ = [basis_embedding(x) for x in x_train_bin]
        x_test_circ = [basis_embedding(x) for x in x_test_bin]
        x_val_circ = [basis_embedding(x) for x in x_val_bin]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        x_val_tfcirc = tfq.convert_to_tensor(x_val_circ)

    if args.embedding == 'angle':
        print(args.embeddingparam, 'Angle embedding!')
        train_maximum = np.max(np.abs(encoded_x_train))
        test_maximum = np.max(np.abs(encoded_x_test))
        val_maximum = np.max(np.abs(encoded_x_val))
        x_train_norm = encoded_x_train / train_maximum
        x_test_norm = encoded_x_test / test_maximum
        x_val_norm = encoded_x_val / val_maximum

        x_train_circ = [angle_embedding(x, args.embeddingparam) for x in x_train_norm]
        x_test_circ = [angle_embedding(x, args.embeddingparam) for x in x_test_norm]
        x_val_circ = [angle_embedding(x, args.embeddingparam) for x in x_val_norm]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        x_val_tfcirc = tfq.convert_to_tensor(x_val_circ)

    if args.embedding == 'bin':
        print('No embedding!')
        x_train_tfcirc = x_train_bin
        x_test_tfcirc = x_test_bin
        x_val_tfcirc = x_val_bin

    if args.embedding == 'no':
        print('No embedding!')
        x_train_tfcirc = encoded_x_train
        x_test_tfcirc = encoded_x_test
        x_val_tfcirc = encoded_x_val

    if args.embedding == None:
        print('Pleaes choose quantum embedding method! basis, angle, no')
        return

    # ----------------------------------------------------------------------------------------------------------------------

    time_3 = time.time()
    passed = time_3 - time_2
    print('Elapsed time for quantum embedding:', passed)

    """MODEL BUILDING"""
    if args.train_layer == 'farhi':
        circuit, readout = create_fvqc(args.observable)

    if args.train_layer == 'grant':
        circuit, readout = create_gvqc(args.observable)

    if args.train_layer == 'dense':
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(4, 4, 1)),
            tf.keras.layers.Dense(2, activation='relu'),
            tf.keras.layers.Dense(1),
        ])

    if args.train_layer != 'farhi' and args.train_layer != 'grant' and args.train_layer != 'dense':
        print('Chose a trainig layer! farhi, grant, dense')
        return

    if args.train_layer == 'farhi' or args.train_layer == 'grant':
        model = tf.keras.Sequential([
            # The input is the data-circuit, encoded as a tf.string
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            # The PQC layer returns the expected value of the readout gate, range [-1,1].
            tfq.layers.PQC(circuit, readout),
        ])

    if args.loss == 'hinge':
        print('Hinge loss selected!')
        model_loss = tf.keras.losses.Hinge()

    if args.loss == 'squarehinge':
        print('Square hinge loss selected!')
        model_loss = tf.keras.losses.SquaredHinge()

    if args.loss == 'crossentropy':
        model_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.0)

    if args.loss == None:
        print('Chose a loss function! hinge, squarehinge')
        return

    if args.optimizer == 'adam':
        model_optimizer = tf.keras.optimizers.Adam()

    if args.optimizer == 'bobyqa':
        model_optimizer = 0

    if args.optimizer == None:
        print('Chose an optimizer!')
        return

    print('Compiling model .....')
    if args.train_layer == 'dense':
        model.compile(
            loss=model_loss,
            optimizer=model_optimizer,
            metrics=['accuracy'])
    if args.train_layer == 'farhi' or args.train_layer == 'grant':
        model.compile(
            loss=model_loss,
            optimizer=model_optimizer,
            metrics=[hinge_accuracy])

    qnn_history = model.fit(
        x_train_tfcirc, y_train,
        batch_size=args.batchsize2,
        epochs=args.epochs,
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
    if args.train_layer == 'farhi' or args.train_layer == 'grant':
        plt.figure(figsize=(10, 5))
        plt.plot(qnn_history.history['hinge_accuracy'], label='Accuracy')
        plt.plot(qnn_history.history['val_hinge_accuracy'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(log_path + '/acc.png')

    if args.train_layer == 'dense':
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

    if args.loss == 'hinge' or args.loss == 'squarehinge':
        # Hinge labels to 0,1
        y_true = (y_true + 1) / 2
        y_pred = (np.array(y_pred) + 1) / 2

        # Round Labels for Metrics
        y_pred_int = []
        for i in range(0, len(y_pred)):
            y_pred_int.append(round(y_pred[i][0]))

    if args.loss == 'crossentropy':
        y_true = tf.squeeze(y_true) > 0.5
        y_pred_int = tf.squeeze(y_pred) > 0.5

    precision_0 = precision_score(y_true, y_pred_int, pos_label=0, average='binary')
    recall_0 = recall_score(y_true, y_pred_int, pos_label=0, average='binary')
    f1_0 = f1_score(y_true, y_pred_int, pos_label=0, average='binary')

    precision_1 = precision_score(y_true, y_pred_int, pos_label=1, average='binary')
    recall_1 = recall_score(y_true, y_pred_int, pos_label=1, average='binary')
    f1_1 = f1_score(y_true, y_pred_int, pos_label=1, average='binary')

    print('Precision for class ', args.class1, ' is: ', precision_0)
    print('Recall for class ', args.class1, ' is: ', recall_0)
    print('F1 for class ', args.class1, ' is: ', f1_0)

    print('Precision for class ', args.class2, ' is: ', precision_1)
    print('Recall for class ', args.class2, ' is: ', recall_1)
    print('F1 for class ', args.class2, ' is: ', f1_1)


# ----------------------------------------------------------------------------------------------------------------------


def extract_features(directory, sample_count, image_size):
    if args.dataset == 'eurosat':
        if args.vgg16 and args.preprocessing != 'ds':
            features = np.zeros(shape=(sample_count, 2, 2, 512))
        if not args.vgg16:
            features = np.zeros(shape=(sample_count, image_size[0], image_size[1], image_size[2]))
    if args.dataset == 'resisc45':
        if args.vgg16 and args.preprocessing != 'ds':
            features = np.zeros(shape=(sample_count, 8, 8, 512))
        if not args.vgg16:
            features = np.zeros(shape=(sample_count, image_size[0], image_size[1], image_size[2]))

    labels_2 = np.zeros(shape=(sample_count, 2))  # how to find label size ?
    labels_3 = np.zeros(shape=(sample_count, 3))  # how to find label size ?

    if args.vgg16 and args.preprocessing != 'ds':
        conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(image_size[0], image_size[1], 3))

    generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(directory,
                                                                         target_size=(image_size[0], image_size[1]),
                                                                         batch_size=args.batchsize1,
                                                                         class_mode='categorical')

    i = 0

    print('Entering for loop...')

    for inputs_batch, labels_batch in generator:

        if args.vgg16 and args.preprocessing != 'ds':
            features_batch = conv_base.predict(inputs_batch)
        if not args.vgg16:
            features_batch = inputs_batch
        if args.preprocessing == 'ds':
            features_batch = inputs_batch

        features[i * args.batchsize1: (i + 1) * args.batchsize1] = features_batch
        try:
            labels_2[i * args.batchsize1: (i + 1) * args.batchsize1] = labels_batch
            labels = labels_2
        except:
            labels_3[i * args.batchsize1: (i + 1) * args.batchsize1] = labels_batch
            labels = labels_3

        i += 1
        if i * args.batchsize1 >= sample_count:
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
    input_test = torch.Tensor(x_binary[0:len_input]).to(args.device)

    transformed, tmp = dae.encode(input_test)

    for i in range(0, len(transformed)):
        encoded_x.append(transformed[i].detach().cpu().numpy())
    encoded_x = np.array(encoded_x).reshape(len_input, 4, 4)

    return encoded_x


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate a hybrid classical-quantum system')

    parser.add_argument('-da', '--dataset', type=str, default='eurosat', help='select dataset. currently available: eurosat, resisc45')

    parser.add_argument('-dp', '--dataset_path', type=str, default='../DATASETS/EuroSAT/2750', help='select dataset path')

    parser.add_argument('-c1', '--class1', type=str, default='AnnualCrop', help='select a class for binary classification')

    parser.add_argument('-c2', '--class2', type=str, default='SeaLake', help='select a class for binary classification')

    parser.add_argument('-ic', '--image_count', type=int, default=3000, help='define number of images')

    parser.add_argument('-b1', '--batchsize1', type=int, default=32, help='batch size for preprocessing')

    parser.add_argument('-b2', '--batchsize2', type=int, default=32, help='batch size for training')

    parser.add_argument('-e', '--epochs', type=int, default=1, help='number of training epochs') # TO DO: CHANGE DEFAULT EPOCHS

    parser.add_argument('-t', '--train_layer', type=str, default='farhi', help='select a training layer. currently available: farhi, grant, dense')

    parser.add_argument('-v', '--vgg16', type=bool, default=False, help='use vgg16 for prior feature extraction True or False')

    parser.add_argument('-cp', '--cparam', type=int, default=0, help='cparam. currently has no influence')

    parser.add_argument('-em', '--embedding', type=str, default='angle', help='select quantum encoding for the classical input data. currently available: basis, angle, ( and bin for no quantum embedding but binarization')

    parser.add_argument('-emp', '--embeddingparam', type=str, default='x', help='select axis for angle embedding')

    parser.add_argument('-l', '--loss', type=str, default='squarehinge', help='select loss function. currently available: hinge, squarehinge, crossentropy')

    parser.add_argument('-ob', '--observable', type=str, default='x', help='select pauli measurement/ quantum observable')

    parser.add_argument('-op', '--optimizer', type=str, default='adam', help='select optimizer. currently available: adam')

    parser.add_argument('-g', '--grayscale', type=bool, default=False, help='transform input to grayscale True or False')

    parser.add_argument('-p', '--preprocessing', type=str, default='pca', help='select preprocessing technique. currently available: ds, pca, fa, ae, dae (=convae if vgg16=False), rbmae')

    parser.add_argument('-de', '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available() ')

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args

if __name__ == "__main__":
    args = parse_args()

    try:
        train(args)
    except FileExistsError:
        print('File already exists!')

