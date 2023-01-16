import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

import pennylane as qml

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

num = 8 # 22 works! 24 did work? 26 oom! 25 is max????

# MNIST data loading

mnist = keras.datasets.mnist

# datasets are numpy.ndarrays
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()     

# normalize the pixels from 0 ~ 255 to 0 ~ 1 
X_train, X_test = X_train / 255.0, X_test / 255.0


# shorten data
X_train = X_train[:5000]
Y_train = Y_train[:5000]
X_test = X_test[:5000]
Y_test = Y_test[:5000]

# get only 2 and 6 for binary classification
X_train_red = []
Y_train_red = []
for i in range(len(X_train)):
    if Y_train[i] == 2 or Y_train[i] == 6:
        X_train_red.append(X_train[i])
        Y_train_red.append(Y_train[i])
        
X_test_red = []
Y_test_red = []
for i in range(len(X_test)):
    if Y_test[i] == 2 or Y_test[i] == 6:
        X_test_red.append(X_test[i])
        Y_test_red.append(Y_test[i])

# List to numpyp.array
X_train = np.asarray(X_train_red)
X_test = np.asarray(X_test_red)
Y_train = np.asarray(Y_train_red)
Y_test = np.asarray(Y_test_red)

# Convert to hinge labels: -1(=2) and 1(=6)
Y_train_ = []
Y_test_ = []

for i in range(len(Y_train)):
    if Y_train[i] == 2:
        Y_train_.append(-1)
    if Y_train[i] == 6:
        Y_train_.append(1)
        
for i in range(len(Y_test)):
    if Y_test[i] == 2:
        Y_test_.append(-1)
    if Y_test[i] == 6:
        Y_test_.append(1)        
        
Y_train = np.asarray(Y_train_)
Y_test = np.asarray(Y_test_)

# Flatten and reshape
X_train_flat = []
for sample in X_train:
    X_train_flat.append(np.ndarray.flatten(sample))
    
X_test_flat = []
for sample in X_test:
    X_test_flat.append(np.ndarray.flatten(sample))

X_train_flat = np.asarray(X_train_flat, dtype=np.float64)
X_test_flat = np.asarray(X_test_flat, dtype=np.float64)

#X_train_flat = X_train_flat[:,0]
#X_test_flat = X_test_flat[:,0]

# Dimensionality reduction from 28x28 to num elements by PCA
pca = PCA(n_components=num)

pca.fit(X_train_flat)

X_train_flat_dr = pca.transform(X_train_flat)
X_test_flat_dr = pca.transform(X_test_flat)

#X_train_dr = X_train_flat_dr.reshape(len(X_train), 4, 4)
#X_test_dr = X_test_flat_dr.reshape(len(X_test), 4, 4)
X_train_dr = X_train_flat_dr
X_test_dr = X_test_flat_dr

# Create PennyLane QML Device
dev = qml.device('qulacs.simulator', wires=num+1, gpu=True)
#dev = qml.device("lightning.gpu", wires=9) # default ist "default.qubit" ---- change to lightning.qubit ???
# For pennylane-lightning[gpu]
# https://docs.pennylane.ai/projects/lightning/en/latest/installation.html

def get_angles(x):
    # Rescale to a range between 0 and pi/2
    values = np.ndarray.flatten(x)
    values = values - min(values)
    values = (values/max(values)) * (np.pi/2)
    values = values.reshape(len(x), num)
    return values.astype('float64')

def statepreparation(x_):
    # State preparation for readout qubit
    qml.PauliX(wires=0)
    qml.Hadamard(wires=0)

    # Simple Y-rotation encoding for classical input data
    for i in range(1, num+1, 1):
        qml.RY(x_[i-1], wires=[i])

def layerXX(W):
    for i in range(0, num, 1):
        qml.IsingXX(phi=W[0,i], wires=[0, i+1], do_queue=True, id=None)
        #qml.CNOT(wires=[0, i+1], do_queue=True, id=None)

def layerZZ(W):
    for i in range(0, num, 1):
        qml.IsingZZ(phi=W[1,i], wires=[0, i+1], do_queue=True, id=None)
        #qml.CNOT(wires=[0, i+1], do_queue=True, id=None)

@qml.qnode(dev, interface='tf') # muss 'diff_method="adjoint"' gesetzt sein? was ist default?
def circuit(inputs, weights): # MUST INCLUDE ARGUMENT WITH NAME: inputs !!!!!!!!!!!
    
    #inputs = get_angles(inputs)

    # Readout preparation and quantum encoding
    #statepreparation(inputs)

    qml.PauliX(wires=0)
    qml.Hadamard(wires=0)

    qml.AngleEmbedding(features=inputs, wires=range(1,num+1), rotation='X')

    # Add layers
    layerXX(weights)
    layerZZ(weights)

    # Hadamard gate on readout qubit for X-basis measurement
    qml.Hadamard(wires=0)

    return qml.expval(qml.PauliZ(wires=0))

def hinge_accuracy(Y, predictions):
    y_true = tf.squeeze(Y) > 0.0
    y_pred = tf.squeeze(predictions) > 0.0
    result = tf.cast(y_true == y_pred, tf.float64)

    return tf.reduce_mean(result)

square_hinge_loss = tf.keras.losses.SquaredHinge()

num_qubits = num+1
num_layers = 2

tf.keras.backend.set_floatx('float64')

opt = tf.keras.optimizers.Adam() # qml.AdamOptimizer()
'''
# ----------------------------------------------------------------------------

cmodel = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(4, 4, 1)),
            tf.keras.layers.Dense(2, activation='relu'),
            tf.keras.layers.Dense(1),
        ])

cmodel.compile(opt, loss=square_hinge_loss, metrics=[hinge_accuracy])

X_train_dr = (X_train_dr.reshape(len(X_train_dr), 4, 4)).astype('float64')
X_test_dr = (X_test_dr.reshape(len(X_test_dr), 4, 4)).astype('float64')
Y_train = Y_train.astype('float64')
Y_test = Y_test.astype('float64')

hybrid = cmodel.fit(X_train_dr, 
                   Y_train,
                   epochs = 15,
                   batch_size = 32,
                   shuffle = True, 
                   validation_data = (X_test_dr, Y_test))
# ----------------------------------------------------------------------------
'''
#weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 1)
#weights = weights_init
# print(weights_init)

#print(qml.draw(circuit)(weights_init, X_train_dr[0]))

# convert the quantum layer to a Keras layer
#shape_tup = weights.shape

weight_shapes = {'weights': (num_layers, num_qubits)}

#print(qml.draw(circuit)(weight_shapes, X_train_dr[0]))


qlayer = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=1, dtype='float64')

# Create keras model
model = tf.keras.models.Sequential([qlayer]) #tf.keras.layers.Dense(8, dtype='float32'),

#model = tf.keras.models.Sequential([tf.keras.layers.Input(shape=(8,), name = "input_0"), qlayer])#, tf.keras.layers.Dense(1, activation='softmax', name = "dense_1")])

model.compile(opt, loss=square_hinge_loss, metrics=[hinge_accuracy])

#X_train_dr = get_angles(X_train_dr)
#X_test_dr = get_angles(X_test_dr)

#X_train_dr = get_angles(X_train_dr)
#X_test_dr = get_angles(X_test_dr)

#model.summary()
X_train_dr = X_train_dr.astype('float64')
X_test_dr = X_test_dr.astype('float64')
Y_train = Y_train.astype('float64')
Y_test = Y_test.astype('float64')

#print(np.unique(X_train_dr))
#print(np.unique(X_test_dr))
#print(np.unique(Y_train))
#print(np.unique(Y_test))

#X_train_dr = tf.cast(X_train_dr, tf.float32)
#X_test_dr = tf.cast(X_test_dr, tf.float32)
#Y_train = tf.cast(Y_train, tf.float32)
#Y_test = tf.cast(Y_test, tf.float32)

hybrid = model.fit(X_train_dr, 
                   Y_train,
                   epochs = 10,
                   batch_size = 16,
                   shuffle = True, 
                   validation_data = (X_test_dr, Y_test))

#if __name__ == '__main__':
#    main()