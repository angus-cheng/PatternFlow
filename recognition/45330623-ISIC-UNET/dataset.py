import torch
import torchvision
import wget
import os
import zipfile

def dir_path_exist(path):
    path = path.replace('\\', '/')
    data_path = os.path.join(path, 'data').replace('\\', '/')

    if os.path.isdir(data_path):
        return True
    elif os.path.isdir(path):
        return True
    else: 
        os.mkdir(path)
        return False

def unzip_data(zip_path, target_dir):
    for item in os.listdir(zip_path):
        exist_dir = zip_path + '/' + item
        if os.path.isdir(exist_dir):
            return
        if item.endswith('.zip'):
            with zipfile.ZipFile(zip_path + '/' + item, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
                zip_ref.close()
    return

def load_data(web_url, path):
    dir = 'data'
    path = os.path.join(path, dir).replace('\\', '/')
    if not dir_path_exist(path):
        wget.download(web_url, out=path)
    unzip_data(path, path)

    return

def preprocess_data(dir, height, width):
    data = torchvision.io.read_file(dir)
    images = torchvision.io.decode_jpeg(data, mode=torch.io.ImageReadMode.RGB) 
    images = torchvision.transforms.Resize(img=images, size=(height, width))
    # Normalize
    images = images / 255.0

    return images

def preprocess_masks(dir, height, width):
    data = torchvision.io.read_file(dir)
    images = torchvision.io.decode_png(data, mode=torch.io.ImageReadMode.GRAY)
    images = torchvision.transforms.Resize(img=images, size=(height, width))
    images = images / 255.0
    # threshold 
    images = (images > 0.5).float()

    return images
    

path = './'
training_data_url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip'
training_ground_truth_url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip'

load_data(training_data_url, path)
load_data(training_ground_truth_url, path)