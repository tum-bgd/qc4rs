import os
import pandas as pd
import shutil
from pathlib import Path


def organize_data(dataset_name, input_path, classes, split):
    image_path = Path(input_path)

    # Get sub directories
    types = os.listdir(image_path)
    print("Categories: ", types)

    # class_1 = types.index(classes[0])
    # class_2 = types.index(classes[1])

    types = [classes[0], classes[1]]
    print("Selected categories: ", types)

    # A list that is going to contain tuples: (type, corresponding image path)
    images = []

    for type in types:
        # Get all the file names
        all_images = os.listdir(image_path / type)
        # Add them to the list
        for image in all_images:
            images.append((type, str(image_path / type) + '/' + image))

    # Build a dataframe
    images = pd.DataFrame(data=images, columns=['category', 'image'], index=None)

    # How many samples for each category are present
    print("Total number of images in the dataset: ", len(images))
    image_count = images['category'].value_counts()
    print("Images in each category: ")
    print(image_count)

    try:
        os.mkdir("../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]))
    except FileExistsError:
        print(str(dataset_name), ' data directory already exists')

    try:
        os.mkdir("../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/train")
        os.mkdir("../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/test")
        os.mkdir("../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/valid")
    except FileExistsError:
        print('train,test,val directories already exist')

    try:
        # Inside the train and validation sub=directories, sub-directories for each catgeory
        os.mkdir(
            "../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/train" + "/" + str(types[0]))
        os.mkdir(
            "../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/train" + "/" + str(types[1]))
        os.mkdir("../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/test" + "/" + str(types[0]))
        os.mkdir("../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/test" + "/" + str(types[1]))
        os.mkdir(
            "../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/valid" + "/" + str(types[0]))
        os.mkdir(
            "../" + str(dataset_name) + "_data_" + str(classes[0]) + "_" + str(classes[1]) + "/valid" + "/" + str(types[1]))
    except FileExistsError:
        print(classes[0], ', ', classes[1], ' directories already exist')
        return

    for category in image_count.index:
        samples = images['image'][images['category'] == category].values
        for i in range(int(split / 2)):
            name = samples[i].split('/')[-1]
            shutil.copyfile(samples[i], './' + "../" + str(dataset_name) + '_data_' + str(classes[0]) + '_' + str(
                classes[1]) + '/test/' + str(category) + '/' + name)
        for i in range(int(split / 2), split):
            name = samples[i].split('/')[-1]
            shutil.copyfile(samples[i], './' + "../" + str(dataset_name) + '_data_' + str(classes[0]) + "_" + str(
                classes[1]) + '/valid/' + str(category) + '/' + name)
        for i in range(split, len(samples)):
            name = samples[i].split('/')[-1]
            shutil.copyfile(samples[i], './' + "../" + str(dataset_name) + '_data_' + str(classes[0]) + '_' + str(
                classes[1]) + '/train/' + str(category) + '/' + name)

    print('Train/Test/Val split and directory creation completed!')


def organize_data_ovr(dataset_name, input_path, classes, split):
    image_path = Path(input_path)

    # Get sub directories
    types = os.listdir(image_path)
    print("Categories: ", types)

    # class_1 = types.index(classes[0])
    # class_2 = types.index(classes[1])

    types = []
    for i in range(len(classes)):
        types.append(classes[i])

    print("Selected categories: ", types)

    # A list that is going to contain tuples: (type, corresponding image path)
    images = []

    for type in types:
        # Get all the file names
        all_images = os.listdir(image_path / type)
        # Add them to the list
        for image in all_images:
            images.append((type, str(image_path / type) + '/' + image))

    # Build a dataframe
    images = pd.DataFrame(data=images, columns=['category', 'image'], index=None)

    # How many samples for each category are present
    print("Total number of images in the dataset: ", len(images))
    image_count = images['category'].value_counts()
    print("Images in each category: ")
    print(image_count)

    splits = image_count * split

    try:
        os.mkdir("../" + str(dataset_name) + "_data_OvR")
    except FileExistsError:
        print(str(dataset_name), ' data directory already exists')

    try:
        os.mkdir("../" + str(dataset_name) + "_data_OvR/train")
        os.mkdir("../" + str(dataset_name) + "_data_OvR/test")
        os.mkdir("../" + str(dataset_name) + "_data_OvR/valid")
    except FileExistsError:
        print('train,test,val directories already exist')

    try:
        # Inside the train and validation sub=directories, sub-directories for each catgeory
        for i in range(len(types)):
            os.mkdir("../" + str(dataset_name) + "_data_OvR/train" + "/" + str(types[i]))
        for i in range(len(types)):
            os.mkdir("../" + str(dataset_name) + "_data_OvR/test" + "/" + str(types[i]))
        for i in range(len(types)):
            os.mkdir("../" + str(dataset_name) + "_data_OvR/valid" + "/" + str(types[i]))
    except FileExistsError:
        print(types[i], ' directory already exist')
        return

    k = 0
    for category in image_count.index:
        samples = images['image'][images['category'] == category].values
        for i in range(int(splits[k] / 2)):
            name = samples[i].split('/')[-1]
            shutil.copyfile(samples[i], './' + "../" + str(dataset_name) + '_data_OvR/test/' + str(category) + '/' + name)
        for i in range(int(splits[k] / 2), int(splits[k])):
            name = samples[i].split('/')[-1]
            shutil.copyfile(samples[i], './' + "../" + str(dataset_name) + '_data_OvR/valid/' + str(category) + '/' + name)
        for i in range(int(splits[k]), len(samples)):
            name = samples[i].split('/')[-1]
            shutil.copyfile(samples[i], './' + "../" + str(dataset_name) + '_data_OvR/train/' + str(category) + '/' + name)
        k += 1

    print('Train/Test/Val split and directory creation for OvR completed!')