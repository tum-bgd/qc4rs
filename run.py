from train import train
import argparse


parser = argparse.ArgumentParser(description='Train and evaluate a hybrid classical-quantum system')

parser.add_argument('-da', '--dataset', type=str, default='eurosat', help='select dataset. currently available: eurosat, resisc45')

parser.add_argument('-dp', '--dataset_path', type=str, default='../2750', help='select dataset path')

parser.add_argument('-c1', '--class1', type=str, default='AnnualCrop', help='select a class for binary classification')

parser.add_argument('-c2', '--class2', type=str, default='SeaLake', help='select a class for binary classification')

parser.add_argument('-ic', '--image_count', type=int, default=3000, help='define number of images')

parser.add_argument('-b1', '--batchsize1', type=int, default=32, help='batch size for preprocessing')

parser.add_argument('-b2', '--batchsize2', type=int, default=32, help='batch size for training')

parser.add_argument('-e', '--epochs', type=int, default=30, help='number of training epochs')

parser.add_argument('-t', '--train_layer', type=str, default='farhi', help='select a training layer. currently available: farhi, grant, dense')

parser.add_argument('-v', '--vgg16', type=bool, default=True, help='use vgg16 for prior feature extraction True or False')

parser.add_argument('-cp', '--cparam', type=int, default=0, help='cparam. currently has no influence')

parser.add_argument('-em', '--embedding', type=str, default='angle', help='select quantum encoding for the classical input data. currently available: basis, angle, ( and bin for no quantum embedding but binarization')

parser.add_argument('-emp', '--embeddingparam', type=str, default='x', help='select axis for angle embedding')

parser.add_argument('-l', '--loss', type=str, default='squarehinge', help='select loss function. currently available: hinge, squarehinge, crossentropy')

parser.add_argument('-ob', '--observable', type=str, default='x', help='select pauli measurement/ quantum observable')

parser.add_argument('-op', '--optimizer', type=str, default='adam', help='select optimizer. currently available: adam')

parser.add_argument('-g', '--grayscale', type=bool, default=False, help='TBD: transform input to grayscale True or False')

parser.add_argument('-p', '--preprocessing', type=str, default='dae', help='select preprocessing technique. currently available: ds, pca, fa, ae, dae (=convae if vgg16=False), rbmae')

parser.add_argument('-de', '--device', type=str, default=None, help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available() ')

'''
fvqc
'''
i = 0
while i<5:
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'fvqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'ds', ])
    train(args) # DS
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'fvqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'pca', ])
    train(args) # PCA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'fvqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'fa', ])
    train(args) # FA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'fvqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'ae', ])
    train(args) # CONVAE
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'fvqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'rbmae', ])
    train(args) # RBMAE
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'fvqc', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'pca', ])
    train(args) # VGG16 + PCA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'fvqc', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'fa', ])
    train(args) # VGG16 + FA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'fvqc', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'ae', ])
    train(args) # VGG16 + AE
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'fvqc', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'rbmae', ])
    train(args) # VGG16 + RBMAE
    i+=1

'''
gvqc
'''
i = 0
while i<5:
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'gvqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'ds', ])
    train(args) # DS
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'gvqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'pca', ])
    train(args) # PCA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'gvqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'fa', ])
    train(args) # FA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'gvqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'ae', ])
    train(args) # CONVAE
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'gvqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'rbmae', ])
    train(args) # RBMAE
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'gvqc', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'pca', ])
    train(args) # VGG16 + PCA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'gvqc', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'fa', ])
    train(args) # VGG16 + FA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'gvqc', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'ae', ])
    train(args) # VGG16 + AE
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'gvqc', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'rbmae', ])
    train(args) # VGG16 + RBMAE
    i+=1

'''
svqc
'''
i = 0
while i<5:
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'svqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'ds', ])
    train(args) # DS
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'svqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'pca', ])
    train(args) # PCA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'svqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'fa', ])
    train(args) # FA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'svqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'ae', ])
    train(args) # CONVAE
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'svqc', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'rbmae', ])
    train(args) # RBMAE
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'svqc', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'pca', ])
    train(args) # VGG16 + PCA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'svqc', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'fa', ])
    train(args) # VGG16 + FA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'svqc', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'ae', ])
    train(args) # VGG16 + AE
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'svqc', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'rbmae', ])
    train(args) # VGG16 + RBMAE
    i+=1

'''
mera
'''
i = 0
while i<5:
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'mera', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'ds', ])
    train(args) # DS
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'mera', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'pca', ])
    train(args) # PCA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'mera', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'fa', ])
    train(args) # FA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'mera', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'ae', ])
    train(args) # CONVAE
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'mera', '-v', False, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'rbmae', ])
    train(args) # RBMAE
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'mera', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'pca', ])
    train(args) # VGG16 + PCA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'mera', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'fa', ])
    train(args) # VGG16 + FA
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'mera', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'ae', ])
    train(args) # VGG16 + AE
    args = parser.parse_args(['-da', 'eurosat', '-dp', '../2750', '-c1', 'AnnualCrop', '-c2', 'SeaLake', '-ic', '3000', '-b1', '32', '-b2', '32','-e', '30', '-t', 'mera', '-v', True, '-em', 'angle', '-emp',  'x', '-l', 'squarehinge', '-ob', 'x', '-op', 'adam', '-g', False, '-p', 'rbmae', ])
    train(args) # VGG16 + RBMAE
    i+=1