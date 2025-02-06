import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def get_mean_stdinv(img):
    """
    Compute the mean and std for input image (make sure it's aligned with training)
    """

    mean=[123.675, 116.28, 103.53]
    std=[58.395, 57.12, 57.375]

    mean_img = np.zeros((img.shape))
    mean_img[:,:,0] = mean[0]
    mean_img[:,:,1] = mean[1]
    mean_img[:,:,2] = mean[2]
    mean_img = np.float32(mean_img)

    std_img = np.zeros((img.shape))
    std_img[:,:,0] = std[0]
    std_img[:,:,1] = std[1]
    std_img[:,:,2] = std[2]
    std_img = np.float64(std_img)

    stdinv_img = 1 / np.float32(std_img)

    return mean_img, stdinv_img

def numpy2tensor(img):
    """
    Convert numpy to tensor
    """
    img = torch.from_numpy(img).transpose(0,2).transpose(1,2).unsqueeze(0).float()
    return img

def prepare_input(img, device):
    """
    Convert numpy image into a normalized tensor (ready to do segmentation)
    """
    mean_img, stdinv_img = get_mean_stdinv(img)
    img_tensor = numpy2tensor(img).to(device)
    mean_img_tensor = numpy2tensor(mean_img).to(device)
    stdinv_img_tensor = numpy2tensor(stdinv_img).to(device)
    img_tensor = img_tensor - mean_img_tensor
    img_tensor = img_tensor * stdinv_img_tensor
    return img_tensor

def draw_text_on_image(image_path, text, scale):
    # Open an image file
    with Image.open(image_path) as img:
        width, height = img.size
        text_size = int(min(width, height) * scale)
        font = ImageFont.truetype("DejaVuSans.ttf", text_size)
        draw = ImageDraw.Draw(img)
        text_position = (20, 0) # horizontal, vertical
        # Add text to image
        draw.text(text_position, text, font=font, fill=(255, 105, 180))

    return img

def find_non_zero_elements(grid):
    non_zero_positions = []
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            if value != 0:
                non_zero_positions.append((i, j))
    return non_zero_positions

def contains_non_zero(patch_row, patch_col):
    patch_start_row = patch_row * patch_size
    patch_end_row = (patch_row + 1) * patch_size - 1
    patch_start_col = patch_col * patch_size
    patch_end_col = (patch_col + 1) * patch_size - 1
    for pos in non_zero_positions:
        if patch_start_row <= pos[0] <= patch_end_row and patch_start_col <= pos[1] <= patch_end_col:
            return True
    return False




if __name__ == '__main__':
    torchscript_file = '../tools/PAL4VST-main/deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt'
    device = 0
    size = 224
    alpha = 0.35
    pink = np.zeros((size, size, 3))
    pink[:, :, 0] = 255
    pink[:, :, 2] = 255

    model = torch.load(torchscript_file).to(device)

    dataset_dir = '../demo/'
    
    
    for file in tqdm(os.listdir(dataset_dir)):
        img_file = dataset_dir+file
        
    # img_file = './example_patch_partition.png'
        img_pil = Image.open(img_file).convert('RGB')
        w, h = img_pil.size[0], img_pil.size[1]

        img = np.array(img_pil.resize((size, size)).convert('RGB'))
        img_tensor = prepare_input(img, device)
        pal = model(img_tensor).cpu().data.numpy()[0][0]  # prediction: Perceptual Artifacts Localization (PAL)
        # print(pal)
        img_with_pal = img * (1 - pal[:, :, None]) + alpha * pink * pal[:, :, None] + (1 - alpha) * img * pal[:, :,
                                                                                                        None]

        non_zero_positions = find_non_zero_elements(pal)

        image = np.array(img_pil)   # cv2.imread('./example_patch_partition.png')

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

        # cv2.imshow('Shuffled Image', shuffled_image)
        shuffled_image = cv2.cvtColor(shuffled_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('../AGIN_patch_partitioned/'+file, shuffled_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

