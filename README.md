# QML4RS
### Quantum Machine Learning for Remote Sensing
Framework for hybrid systems for the classification of satellite imagery. Pre- and postprocessing are performed classical while a parameterized quantum circuit is used for classification. The hybrid systems use 16 data qubits and the quantum systems are simulated on the GPU. Classifications were performed with EuroSAT and NWPU-RESISC45 data but the framework can also be used for arbitrary datasets with small changes.

![General scheme for hybrid systems](Images/systemscheme.PNG)
*Figure 1: General scheme for hybrid systems*

Available preprocessing methods:
* VGG16 (without fully connected top layers for classification)
* Downscaling
* Principal component analysis
* Factor analysis
* Simple autoencoder
* Deep autoencoder
* Convolutional autoencoder
* Autoencoder created from restricted Boltzmann machines
* TBC

Available quantum encodings:
* Basis encoding
* Angle encoding
* TBC

Available quantum circuit architectures:
* FPQC
* GPQC
* TBC

Available loss functions:
* Hinge loss
* Square hinge loss
* Binary cross-entropy loss

Available optimizers:
* Adam
* TBC

The framework can be easily extended with additional preprocessing methods, circuit architectures, etc..

## Setup
Create a directory:
```bash
mkdir qml4rs_dir && cd qml4rs_dir
```
Clone the repository:
```bash
git clone https://github.com/tumbgd/qml4rs.git
```
Download and unzip a dataset, e.g. EuroSAT:
```bash
curl https://madm.dfki.de/files/sentinel/EuroSAT.zip -o EuroSAT.zip && unzip EuroSAT.zip
```
Build the container (The version in the keyring URL inside the Dockerfile is important, note this example is for Debian 10. You can find your version with the command 'lsb_release -a' in a workspace terminal):
```bash
cd qml4rs && docker build -t qml4rs .
```
Start the container:
```bash
docker run -it --rm -v /path/to/qml4rs_dir:/tf --gpus=all --name=qml4rs_container qml4rs
```

## Default usage
Default training and evaluation of a hybrid system with EuroSAT data. By default, binary classification of the EuroSAT classes AnnualCrop and SeaLake is performed. The Dimensionality reduction is performed with a VGG16 combined with a deep autoencoder and the compressed data is angle encoded and classified by a parameterized quantum circuit with the FVQC as training layer. Furthermore, square hinge loss and the Adam optimizer are used by default. 
```bash
python train.py
```
The training layer and all other parts can be easily adapted by several parameters. For a description of the parameter usage check:
```bash
python train.py --help
```
One-versus-rest multiclass classification can be performed. However, the script is currently limited since only classification of the EuroSAT dataset with preprocessing by a VGG16 combined with a deep autoencoder and the FPQC for classification can be performed with sufficent accuracy.
For multiclass classification execute:
```bash
python train_ovr.py
```

## Acknowledgements
* FPQC [Farhi et al., 2018](https://arxiv.org/abs/1802.06002)
* GPQC [Grant et al., 2018](https://arxiv.org/abs/1804.03680)
* VGG16 [Simonyan et al., 2014](https://arxiv.org/abs/1409.1556v5)
* EuroSAT [Helber et al., 2019](https://arxiv.org/abs/1709.00029)
* NWPU-RESISC45 [Cheng et al., 2017](https://arxiv.org/abs/1703.00121)
* Original RBM implementation [eugenet12, 2020](https://github.com/eugenet12/pytorch-rbm-autoencoder)
 
## TBD
* New rbmAE implementation with TensorFlow
* Replace build-in VGG16 with vgg16wFT.py
* Add: optimization methods, preprocessing methods, circuit architectures
* Implement probability calibrated classifiers
