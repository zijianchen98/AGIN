from utils import *
from skimage.io import imsave
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import glob
import pdb
import os

device = 0
size = 512
torchscript_file = './deployment/pal4vst/swin-large_upernet_unified_512x512/end2end.pt'
img_dir = 'F:/PHD/AIGC/AIGN_Database/temp/original'
alpha = 0.5
pink = np.zeros((size, size, 3))
pink[:, :, 0] = 255
pink[:, :, 2] = 255

model = torch.load(torchscript_file).to(device)

# for task_dir in glob.glob(img_dir + '/*'):
#     task = task_dir.split('/')[-1]
#     print(task)
#     vis_dir = 'F:/PHD/AIGC/AIGN_Database/temp/vis_pal'
#     os.makedirs(vis_dir, exist_ok=True)
#     print('Processing: ', task)
#     for img_file in tqdm(glob.glob(os.path.join(task_dir, 'images') + '/*')):
#         fname = os.path.basename(img_file)
#         img_pil = Image.open(img_file)
#         w, h = img_pil.size[0], img_pil.size[1]
#         img = np.array(img_pil.resize((size, size)).convert('RGB'))
#         img_tensor = prepare_input(img, device)
#         pal = model(img_tensor).cpu().data.numpy()[0][0]  # prediction: Perceptual Artifacts Localization (PAL)
#         img_with_pal = img * (1 - pal[:, :, None]) + alpha * pink * pal[:, :, None] + (1 - alpha) * img * pal[:, :,
#                                                                                                           None]
#         Image.fromarray(img_with_pal.astype(np.uint8)).resize((w, h)).save(os.path.join(vis_dir, fname))

vis_dir = 'F:/PHD/AIGC/AIGN_Database/temp/vis_pal_without_red'

for img_file in tqdm(glob.glob(img_dir + '/*')):
    print(img_file)
    fname = os.path.basename(img_file)
    img_pil = Image.open(img_file)
    w, h = img_pil.size[0], img_pil.size[1]
    img = np.array(img_pil.resize((size, size)).convert('RGB'))
    img_tensor = prepare_input(img, device)
    pal = model(img_tensor).cpu().data.numpy()[0][0]  # prediction: Perceptual Artifacts Localization (PAL)
    print(pal)
    img_with_pal = img * (1 - pal[:, :, None]) + alpha * pink * pal[:, :, None] + (1 - alpha) * img * pal[:, :,
                                                                                                      None]

    #with pink region img_with_pal = img * (1 - pal[:, :, None]) + alpha * pink * pal[:, :, None] + (1 - alpha) * img * pal[:, :,
    #                                                                                                       None]

    Image.fromarray(img_with_pal.astype(np.uint8)).resize((w, h)).save(os.path.join(vis_dir, fname))
