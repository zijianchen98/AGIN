import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Subset
import cv2
from utils.AmbrosioTortorelliMinimizer import *
import torch.nn as nn
from utils.patch_partition import prepare_input, find_non_zero_elements, contains_non_zero
from PIL import Image, ImageDraw, ImageFont
import random
import os
from torchvision import transforms



class AGIN_dataloader(Dataset):
    def __init__(self, data_dir, csv_path, transform, database, seed=0):
        self.database = database

        tmp_df = pd.read_csv(csv_path)
        image_name = tmp_df['filename'].to_list()
        mos_N = tmp_df['MOS_nat'].to_list()
        mos_T = tmp_df['MOS_tech'].to_list()
        mos_R = tmp_df['MOS_rat'].to_list()


        n_images = len(image_name)
        random.seed(seed)
        np.random.seed(seed)
        index_rd = np.random.permutation(n_images)


        if 'train' in database:
            index_subset = index_rd[ : int(n_images * 0.7)]
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos_N[i] for i in index_subset]
            self.Y_train_tech = [mos_T[i] for i in index_subset]
            self.Y_train_rat = [mos_R[i] for i in index_subset]
        elif 'test' in database:
            index_subset = index_rd[int(n_images * 0.7) : ]
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos_N[i] for i in index_subset]
            self.Y_train_tech = [mos_T[i] for i in index_subset]
            self.Y_train_rat = [mos_R[i] for i in index_subset]
        elif 'all' in database:
            index_subset = index_rd
            self.X_train = [image_name[i] for i in index_subset]
            self.Y_train = [mos_N[i] for i in index_subset]
            self.Y_train_tech = [mos_T[i] for i in index_subset]
            self.Y_train_rat = [mos_R[i] for i in index_subset]
        else:
            raise ValueError(f"Unsupported subset database name: {database}")
        print(self.X_train)

        self.data_dir = data_dir

        self.transform_rationality = transform

        self.transform_distortion = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform_distortion_preprocessing = transforms.Compose([transforms.ToTensor()])
       
        self.length = len(self.X_train)

    def __getitem__(self, index):
        path = os.path.join(self.data_dir, self.X_train[index])
        path_patch = os.path.join(self.data_dir[:-7]+'partitioned', self.X_train[index])
        img = Image.open(path).convert('RGB')


        img_patch = Image.open(path_patch).convert('RGB')
        
        img_technical = self.transform_distortion_preprocessing(img_patch)

        img_rationality = LFM(self.transform_rationality(img))
        

        y_mos = self.Y_train[index]
        y_mos_tech = self.Y_train_tech[index]
        y_mos_rat = self.Y_train_rat[index]

        y_label = torch.FloatTensor(np.array(float(y_mos)))
        y_label_tech = torch.FloatTensor(np.array(float(y_mos_tech)))
        y_label_rat = torch.FloatTensor(np.array(float(y_mos_rat)))
        
        data = {'img_rationality': img_rationality,
                'img_technical': img_technical,
                'y_label': y_label,
                'y_label_tech':y_label_tech,
                'y_label_rat':y_label_rat}

        return data


    def __len__(self):
        return self.length



def LFM(image):
    result = []
    # img = Image.open(path)
    img = np.array(image)
    
    for channel in cv2.split(img):
        solver = AmbrosioTortorelliMinimizer(channel, iterations=1, tol=0.1,
                                             solver_maxiterations=6)
        f, v = solver.minimize()
        result.append(f)
    
    f = cv2.merge(result)
    img2 = f * 1
    cv2.normalize(img2, img2, 0, 255, cv2.NORM_MINMAX)
    img3 = np.float32(img2)   # uint8
    return img3


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=0)
    correct = torch.eq(preds, labels).cpu().float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc


def binary_acc(outputs, labels):
    preds = torch.round(torch.sigmoid(outputs))  
    correct = torch.eq(preds, labels).float()
    acc = correct.sum() / len(correct)
    return acc


def normalize_list(lst):
    arr = np.array(lst)
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return list(normalized_arr)


def scale_list(input_list):
    min_value = min(input_list)
    max_value = max(input_list)

    scale_factor = (125 - 100) / (max_value - min_value)

    scaled_list = [100 + (x - min_value) * scale_factor for x in input_list]

    return scaled_list



def partition(img_path):
    torchscript_file = './tools/PAL4VST-main/deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt'
    device = 0
    size = 224
    # alpha = 0.35
    pink = np.zeros((size, size, 3))
    pink[:, :, 0] = 255
    pink[:, :, 2] = 255

    model_artifact_detect = torch.load(torchscript_file).to(device)
    img_pil = Image.open(img_path)
    # w, h = img_pil.size[0], img_pil.size[1]

    img = np.array(img_pil.resize((size, size)).convert('RGB'))
    img_tensor = prepare_input(img, device)
    pal = model_artifact_detect(img_tensor).cpu().data.numpy()[0][0]  # prediction: Perceptual Artifacts Localization (PAL)
    # print(pal)

    image = cv2.imread(img_path)

    resized_image = cv2.resize(image, (224, 224))

    patch_size = 32

    num_patches_per_row = resized_image.shape[1] // patch_size
    num_patches_per_col = resized_image.shape[0] // patch_size

    patches = np.zeros((num_patches_per_col, num_patches_per_row, patch_size, patch_size, 3), dtype=np.uint8)
    swapped = np.zeros((num_patches_per_col, num_patches_per_row), dtype=bool)

    for i in range(num_patches_per_col):
        for j in range(num_patches_per_row):
            patches[i, j] = resized_image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]

    for i in range(num_patches_per_col):
        for j in range(num_patches_per_row):
            if not swapped[i, j] and not contains_non_zero(i, j):
                neighbor_positions = [
                    (i - 1, j - 1), (i - 1, j), (i - 1, j + 1),
                    (i, j - 1), (i, j + 1),
                    (i + 1, j - 1), (i + 1, j), (i + 1, j + 1)
                ]
                valid_neighbors = [(ni, nj) for ni, nj in neighbor_positions if
                                   0 <= ni < num_patches_per_col and 0 <= nj < num_patches_per_row and not swapped[
                                       ni, nj] and not contains_non_zero(ni, nj)]
                if valid_neighbors:
                    ni, nj = random.choice(valid_neighbors)
                    patches[i, j], patches[ni, nj] = patches[ni, nj].copy(), patches[i, j].copy()
                    swapped[i, j], swapped[ni, nj] = True, True

    shuffled_image = np.zeros_like(resized_image)
    for i in range(num_patches_per_col):
        for j in range(num_patches_per_row):
            shuffled_image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = patches[i, j]

    return shuffled_image