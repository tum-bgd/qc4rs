'''
TO DO:
1
Set up new docker container with upgraded tensorflow and pennylane
    - get tensorflow docker image, install pennylane, then check dependencies
    - get gpu support with qulacs
    - run test script for validation

2
Implement circuits with pennylane (thus get rid of tfq and cirq)

3
Implement rbmAE with tensorflow and get rid of torch

4
Check what other packages can be deleted

5
update logging
'''
# Standard python librarys
import os
import sys
import time
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

# Core librarys
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.callbacks import CSVLogger # this may be removed

from skimage.transform import downscale_local_mean
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow_quantum as tfq # this will be removed and replaced by pennylane

import preprocessing
#from preprocessing.autoencoderModels import *
#from preprocessing.dae import DAE
#from preprocessing.rbm import train_rbm
import circuits
from utils import *


def train(**kwargs):

    # name = kwargs['name']['value']
    ds = kwargs['dataset']['value']
    ds_path = kwargs['dataset_path']['value']
    im_num = kwargs['image_num']['value']
    im_size = kwargs['image_size']['value']
    c1, c2 = kwargs['classes']['value']
    gray = kwargs['grayscale']['value']
    pp = kwargs['preprocessing']['value']
    vgg = kwargs['vgg16']['value']
    bs_pp = kwargs['batchsize_pre']['value']
    # epochs_pp = kwargs['epochs_pre']['value']
    train_layer = kwargs['train_layer']['value']
    emb = kwargs['embedding']['value']
    emb_param = kwargs['embedding_param']['value']
    obs = kwargs['observable']['value']
    bs_train = kwargs['batchsize_train']['value']
    epochs_train = kwargs['epochs_train']['value']
    loss = kwargs['loss']['value']
    # metric = kwargs['metric']['value']
    optim = kwargs['optimizer']['value']
    # optim_lr = kwargs['optimizer_lr']['value']
    device = kwargs['device']['value']
    if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 16 # equals number of data qubits

    '''
    Logging
    '''
    try:
        os.mkdir('./logs')
    except FileExistsError:
        print('Log directory exists!')

    log_path = os.path.join('./logs/RUN_' + str(ds) + '_' + str(c1) + 'vs' + str(c2) + '_' + str(pp) + '_' + 'vgg16' + str(vgg) + '_' + str(emb) + str(emb_param) + '_' + str(train_layer) + '_' + str(loss) + '_' + str(obs))
    k = 0
    try:
        os.mkdir(log_path)
    except FileExistsError:
        while os.path.exists(log_path):
            log_path = os.path.join('./logs/RUN_' + str(ds) + '_' + str(c1) + 'vs' + str(c2) + '_' + str(pp) + '_' + 'vgg16' + str(vgg) + '_' + str(emb) + str(emb_param) + '_' + str(train_layer) + '_' + str(loss) + '_' + str(obs) + '_' + str(k))
            k+=1
        os.mkdir(log_path)

    sys.stdout = open(log_path + '/output_log.txt', 'w')
    csv_logger = CSVLogger(log_path + '/model_log.csv', append=True, separator=';')

    start = time.time()
    print('OA timer started at:', start)

    organize_data(dataset_name=ds, input_path=ds_path, classes=[c1, c2], split=int(0.3*im_num))

    base_dir = './' + '../' + ds + '_data_' + c1 + '_' + c2
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'valid')

    train_count = im_num * 2 - int(0.6*im_num)
    test_count = int(0.3 * im_num)
    val_count = test_count

    train_features, train_labels = extract_features(ds, train_dir, train_count, im_size, pp, vgg, bs_pp)
    test_features, test_labels = extract_features(ds, test_dir, test_count, im_size, pp, vgg, bs_pp)
    val_features, val_labels = extract_features(ds, val_dir, val_count, im_size, pp, vgg, bs_pp)

    print('Total Number of ' + str(c1) + ' and ' + str(c2) + ' TRAIN images is:' +
          str(len(train_features)))
    print('Total Number of ' + str(c1) + ' and ' + str(c2) + ' TEST images is:' +
          str(len(test_features)))
    print('Total Number of ' + str(c1) + ' and ' + str(c2) + ' VALIDATION images is:' +
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
        # convert labels from 1, 0 to 1, -1
        y_train = 2.0 * y_train - 1.0
        y_test = 2.0 * y_test - 1.0
        y_val = 2.0 * y_val - 1.0

    time_1 = time.time()
    passed = time_1 - start
    print('Elapsed time for preperation:', passed)

    '''
    Dimensionality reduction
    '''

    """GRAYSCALE"""
    if gray and pp != 'ds':
        print('Images BRG2GRAY')
        x_train = []
        x_test = []
        x_val = []

        k = 0
        for img in train_features:
            x_train.append(0.2125 * np.float32(img[:,:,0]) + 0.7154 * np.float32(img[:,:,1]) + 0.0721 * np.float32(img[:,:,2]))
            k += 1
        k = 0
        for img in test_features:
            x_test.append(0.2125 * np.float32(img[:,:,0]) + 0.7154 * np.float32(img[:,:,1]) + 0.0721 * np.float32(img[:,:,2]))
            k += 1
        k = 0
        for img in val_features:
            x_val.append(0.2125 * np.float32(img[:,:,0]) + 0.7154 * np.float32(img[:,:,1]) + 0.0721 * np.float32(img[:,:,2]))
            k += 1

        train_features = np.asarray(x_train)
        test_features = np.asarray(x_test)
        val_features = np.asarray(x_val)

    """DOWNSAMPLING"""
    if pp == 'ds':
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
            x_train[k] = 0.2125 * np.float32(img[:,:,0]) + 0.7154 * np.float32(img[:,:,1]) + 0.0721 * np.float32(img[:,:,2])
            k += 1
        k = 0
        for img in test_features:
            x_test[k] = 0.2125 * np.float32(img[:,:,0]) + 0.7154 * np.float32(img[:,:,1]) + 0.0721 * np.float32(img[:,:,2])
            k += 1
        k = 0
        for img in val_features:
            x_val[k] = 0.2125 * np.float32(img[:,:,0]) + 0.7154 * np.float32(img[:,:,1]) + 0.0721 * np.float32(img[:,:,2])
            k += 1

        # Downsampling
        # https://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html
        ds_param = int(im_size[0] / 4)

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
    if pp == 'pca':
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
    if pp == 'ae':
        x_train, x_test, x_val = flatten_gray_data(train_features, test_features, val_features, train_count, test_count,
                                                   val_count)

        autoencoder = preprocessing.autoencoderModels.SimpleAutoencoder_64(latent_dim)

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

    if pp == 'dae':
        if vgg:
            print('Starting dimensional reduction with VGG16 and autoencoder!')

            if ds == 'eurosat':
                x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count,
                                                      test_count,
                                                      val_count)
                autoencoder = preprocessing.autoencoderModels.DeepAutoencoder_64(latent_dim)

            if ds == 'resisc45':
                x_train, x_test, x_val = flatten_data(train_features, test_features, val_features, train_count,
                                                      test_count,
                                                      val_count)
                autoencoder = preprocessing.autoencoderModels.SimpleAutoencoder_256(latent_dim)

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

        if not vgg:
            print('Starting dimensional reduction with convolutional autoencoder!')

            x_train = train_features.reshape(-1, im_size[0], im_size[1], im_size[2])
            x_test = test_features.reshape(-1, im_size[0], im_size[1], im_size[2])
            x_val = val_features.reshape(-1, im_size[0], im_size[1], im_size[2])

            if im_size[0] == 256:
                autoencoder = preprocessing.autoencoderModels.ConvAutoencoder_256(latent_dim, im_size)

            if im_size[0] == 64:
                autoencoder = preprocessing.autoencoderModels.ConvAutoencoder_64(latent_dim, im_size)

            if im_size[0] != 256 and im_size[0] != 64:
                print('No matching autoencoder for image size' + str(im_size[0]) + 'found!')

            autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

            autoencoder.fit(x_train, x_train,
                            batch_size=bs_pp,
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

    '''
    Here was the RBM Autoencoder model before TO DO: IMPLEMENT NEW ONE
    '''

    if pp == 'fa':
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

    if pp == None:
        print('Please chose a dimensional reduction method! ds, pca, ae, dae')
        return

    time_2 = time.time()
    passed = time_2 - time_1
    print('Elapsed time for data compression:', passed)

    enc_x_train_u = unique2D_subarray(encoded_x_train)
    enc_x_test_u = unique2D_subarray(encoded_x_test)
    enc_x_val_u = unique2D_subarray(encoded_x_val)
    print("Encoded unique arrays: Train", enc_x_train_u.shape, "and: Test", enc_x_test_u.shape, "and: Val",
          enc_x_val_u.shape)


    '''
    Quantum embedding
    '''

    if emb == 'basis' or emb == 'bin':
        x_train_bin, x_test_bin, x_val_bin = binarization(encoded_x_train, encoded_x_test, encoded_x_val)

        """CHECK HOW MANY UNIQUE ARRAYS ARE LEFT AFTER ENCODING"""
        x_train_u = unique2D_subarray(x_train_bin)
        x_test_u = unique2D_subarray(x_test_bin)
        x_val_u = unique2D_subarray(x_val_bin)
        print("Unique arrays after thresholding: Train", x_train_u.shape, "and: Test", x_test_u.shape, "and: Val",
              x_val_u.shape)

    if emb == 'basis':
        print('Basis embedding!')
        x_train_circ = [circuits.embeddings.basis_embedding(x) for x in x_train_bin]
        x_test_circ = [circuits.embeddings.basis_embedding(x) for x in x_test_bin]
        x_val_circ = [circuits.embeddings.basis_embedding(x) for x in x_val_bin]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        x_val_tfcirc = tfq.convert_to_tensor(x_val_circ)

    if emb == 'angle':
        print(emb_param, 'Angle embedding!')
        train_maximum = np.max(np.abs(encoded_x_train))
        test_maximum = np.max(np.abs(encoded_x_test))
        val_maximum = np.max(np.abs(encoded_x_val))
        x_train_norm = encoded_x_train / train_maximum
        x_test_norm = encoded_x_test / test_maximum
        x_val_norm = encoded_x_val / val_maximum

        x_train_circ = [circuits.embeddings.angle_embedding(x, emb_param) for x in x_train_norm]
        x_test_circ = [circuits.embeddings.angle_embedding(x, emb_param) for x in x_test_norm]
        x_val_circ = [circuits.embeddings.angle_embedding(x, emb_param) for x in x_val_norm]
        x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
        x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)
        x_val_tfcirc = tfq.convert_to_tensor(x_val_circ)

    if emb == 'bin':
        print('No embedding!')
        x_train_tfcirc = x_train_bin
        x_test_tfcirc = x_test_bin
        x_val_tfcirc = x_val_bin

    if emb == 'none':
        print('No embedding!')
        x_train_tfcirc = encoded_x_train
        x_test_tfcirc = encoded_x_test
        x_val_tfcirc = encoded_x_val

    if emb == None:
        print('Pleaes choose quantum embedding method! basis, angle, none')
        return

    time_3 = time.time()
    passed = time_3 - time_2
    print('Elapsed time for quantum embedding:', passed)

    '''
    Model building and training
    '''


    if train_layer == 'fvqc':
        circuit, readout = circuits.fvqc.create_fvqc(obs)

    if train_layer == 'gvqc':
        circuit, readout = circuits.gvqc.create_gvqc(obs)

    if train_layer == 'mps':
        circuit, readout = circuits.mps.create_mps(obs)

    if train_layer == 'mera':
        circuit, readout = circuits.mera.create_mera(obs)

    if train_layer == 'svqc':
        circuit, readout = circuits.svqc.create_svqc(obs)

    if train_layer == 'hvqc':
        circuit, readout = circuits.hvqc.create_hvqc(obs)

    if train_layer != 'dense':
        print(circuit)

    if train_layer == 'dense':
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(4, 4, 1)),
            tf.keras.layers.Dense(2, activation='relu'),
            tf.keras.layers.Dense(1),
        ])

    if train_layer == None:
        print('Chose a trainig layer!')
        return

    if train_layer != 'dense':
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

    if loss == None:
        print('Chose a loss function! hinge, squarehinge')
        return

    if optim == 'adam':
        model_optimizer = tf.keras.optimizers.Adam()

    if optim == 'bobyqa':
        model_optimizer = 0

    if optim == None:
        print('Chose an optimizer!')
        return

    print('Compiling model...')
    if train_layer == 'dense':
        model.compile(
            loss=model_loss,
            optimizer=model_optimizer,
            metrics=['accuracy'])
    if train_layer != 'dense':
        model.compile(
            loss=model_loss,
            optimizer=model_optimizer,
            metrics=[hinge_accuracy])

    qnn_history = model.fit(
        x_train_tfcirc, y_train,
        batch_size=bs_train,
        epochs=epochs_train,
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

    # save figures for accuracy and loss
    if train_layer != 'dense':
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

    # model.save(log_path + '/model.h5') is not implemented: https://github.com/tensorflow/quantum/issues/56
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

    print('Precision for class ', c1, ' is: ', precision_0)
    print('Recall for class ', c1, ' is: ', recall_0)
    print('F1 for class ', c1, ' is: ', f1_0)

    print('Precision for class ', c2, ' is: ', precision_1)
    print('Recall for class ', c2, ' is: ', recall_1)
    print('F1 for class ', c2, ' is: ', f1_1)
