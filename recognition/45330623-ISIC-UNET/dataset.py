import torch
import torchvision
import wget
import os
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def download_data(web_url, path):
    dir = 'data'
    path = os.path.join(path, dir).replace('\\', '/')
    if not dir_path_exist(path):
        wget.download(web_url, out=path)
    unzip_data(path, path)

    return

def create_df(path):
    df = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.split('.')[1] != 'csv':
                df.append(filename.split('.')[0])
    
    return pd.DataFrame({'id': df}, index = np.arange(0, len(df)))

path = './'
training_data_url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip'
training_ground_truth_url = 'https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip'

download_data(training_data_url, path)
download_data(training_ground_truth_url, path)

train_dir = './data/ISIC-2017_Training_Data'
mask_dir = './data/ISIC-2017_Training_Part1_GroundTruth'
annotations_file = './data/ISIC-2017_Training_Data/ISIC-2017_Training_Data_metadata.csv'

df = create_df(train_dir)
print(len(df))