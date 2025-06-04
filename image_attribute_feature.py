import cv2
import numpy as np
import os
import pandas as pd


def calculate_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def calculate_colorfulness(image):
    (B, G, R) = cv2.split(image)
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    rg_mean, rg_std = np.mean(rg), np.std(rg)
    yb_mean, yb_std = np.mean(yb), np.std(yb)

    return np.sqrt(rg_std ** 2 + yb_std ** 2) + 0.3 * np.sqrt(rg_mean ** 2 + yb_mean ** 2)

def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def calculate_si(image): # Spatial Information
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Sobel filter in X and Y direction
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Magnitude of gradient
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.mean(magnitude)

