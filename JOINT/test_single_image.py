import os, argparse
import numpy as np
import torch
from torchvision import transforms
from utils.AmbrosioTortorelliMinimizer import *
import models.JOINT as JOINT
from PIL import Image
import cv2
from scipy.optimize import curve_fit
from scipy import stats
import pandas as pd
import random

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


def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="Image Naturalness Assessment")
    parser.add_argument('--model_path', help='Path of model snapshot.', type=str,default='./savedir/JOINT_2024.pth')
    parser.add_argument('--popt_file', type=str, default='./savedir/JOINT_2024.pth')
    parser.add_argument('--model', type=str,default='JOINT')
    parser.add_argument('--image_path', type=str, default='./demo/CoCosNet_23.png')
    parser.add_argument('--resize', type=int, default=384)
    parser.add_argument('--crop_size', help='crop_size.',type=int, default=224)
    parser.add_argument('--gpu_ids', type=list, default=0)


    args = parser.parse_args()

    return args


if __name__ == '__main__':

    random_seed = 2024
    torch.manual_seed(random_seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    args = parse_args()

    model_path = args.model_path
    popt_file = args.popt_file


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load the network
    if args.model == 'JOINT':
        model = JOINT.JOINT_Model()

    
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    popt = np.load(popt_file)
    model.eval()

    
    transform_rationality = transforms.Compose([transforms.Resize(args.resize),
                                               transforms.CenterCrop(args.crop_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    



    transform_technical = transforms.Compose([transforms.Resize(args.resize),
                                               transforms.CenterCrop(args.crop_size),
                                               transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    test_image = Image.open(args.image_path).convert('RGB')
    
    image_path_tech = args.image_path.replace("demo", "AGIN_patch_partitioned")
   
    
    Img_pil_tech = Image.open(image_path_tech).convert('RGB')
    test_image_technical = transform_technical(Img_pil_tech)

    test_image_rationality = transform_rationality(Image.fromarray(np.uint8(LFM(test_image))))

    test_image_technical = test_image_technical.unsqueeze(0)
    test_image_rationality = test_image_rationality.unsqueeze(0)

    with torch.no_grad():
        test_image_technical = test_image_technical.to(device)
        test_image_rationality = test_image_rationality.to(device)
        score_T, score_R = model(test_image_technical, test_image_rationality)
        
       
        print('The technical score of the test image is {:.4f}'.format(score_T.item()))
        print('The rationality score of the test image is {:.4f}'.format(score_R.item()))
        
        score_N = 0.145*score_T+0.769*score_R
        print('The naturalness score of the test image is {:.4f}'.format(score_N.item()))