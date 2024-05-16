#!/usr/bin/env python
# coding: utf-8

import math
import os
from pathlib import Path
import argparse

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from scipy.signal import gaussian
from scipy.signal import argrelextrema
from PIL import Image
from PIL.TiffTags import TAGS

from scripts.utils.utils import *

def get_hist_fg(img):
    """
    This function computes the foreground mask for a single unprocessed dataset image.

    Parameters:
    img (np.ndarray): A unprocessed image from the dataset.

    Returns:
    mask (np.ndarray): Foreground mask (excludes air pixels).
    hist (np.ndarray): Histogram of the image.
    start (int): Foreground threshold pixel intensity.
    """
    # Gaussian blur to reduce image noise
    blurred_img = cv2.GaussianBlur(img, (7, 7), 0)

    # Calculate histogram of image
    hist = cv2.calcHist([blurred_img], [0], None, [256], [0,256])
    hist = np.squeeze(hist)
    hist[0] = 0

    # Find the pixel intensity (x-value) corresponding to the highest peak
    # The pixel-intensities falling under this peak correspond to air
    peak = np.argmax(hist)
    
    # Heuristic to find the pixel-intensity corresponding to the right-side base of the peak
    start = peak + 3
    derivative = np.gradient(hist)
    slope = derivative[start]
    
    for i in range(peak + 3, 256):
        if derivative[i] / slope < 0.025:
            break
        start += 1

    # Morphological operations to remove noise and enhance the foreground
    mask = np.where(img > start, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))

    return mask, hist, start
    

def histogram_thresh(path):
    """
    This is a driver function to compute the foreground of unprocessed dataset images using get_hist_fg().
    This function also computes the average proportion of air pixels in the dataset.

    Parameters:
    path (string): Dataset path.

    Returns:
    pd.DataFrame: Creates a pd.DataFrame with a column containing the proportion of air pixels, indexed by the dataset name.
    """
    air_pixels_proportion_dict = {}
    air_pixels_proportion_list = []

    tiff_file_path =  path + '/tiff'
    tiff_files = get_tiff_dict(tiff_file_path)

    curr_dir_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    hist_solution_path = curr_dir_path + f'../output/{path.split(os.path.sep)[-1]}/hist_threshold/'
    
    # Create the output directory if it doesn't exist already
    if not os.path.isdir(hist_solution_path):
        os.makedirs(hist_solution_path)

    if not os.path.isdir(f'{hist_solution_path}/tiff/'):
        os.makedirs(f'{hist_solution_path}/tiff/')

    tiff_template = Image.open(tiff_files[tiff_file_path][0])
    temp_img = cv2.imread(tiff_files[tiff_file_path][0], cv2.IMREAD_GRAYSCALE)
    out = cv2.VideoWriter(hist_solution_path + path.split('/')[-1] + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (temp_img.shape[1], temp_img.shape[0]))

    print('Begin computing foreground of ' + path)
    for file in tqdm(sorted(tiff_files[tiff_file_path])):
        # Read a dataset image
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # Call the histogram foreground thresholding function
        mask, hist, thresh = get_hist_fg(img)
        masked_image = cv2.bitwise_and(img, mask)

        ### Compute the average proportion of air pixels in the dataset
        total_pixels = img.shape[0] * img.shape[1]
        black_pixels = img[img == 0].shape[0]
        foreground_pixels = mask[mask > 0].shape[0]
        air_pixels = total_pixels - black_pixels - foreground_pixels
        air_pixels_proportion = air_pixels / (total_pixels - black_pixels)
        air_pixels_proportion_list.append(air_pixels_proportion)

        # Save the thresholded image as a TIFF image
        tiff_image = cv2.imread(file, -1)
        tiff_masked = cv2.bitwise_and(tiff_image, (mask > 0).astype(np.uint16) * 0xffff)
        write_tiff(tiff_template, f'{hist_solution_path}/tiff/{file.split(os.path.sep)[-1]}', tiff_masked)

        # Add the thresholded image into the dataset video
        out.write(cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR))
    print('Finish computing foreground of ' + path)

    air_pixels_proportion_dict[path.split(os.path.sep)[-1]] = np.mean(air_pixels_proportion_list)
    out.release()

    return pd.DataFrame.from_dict(air_pixels_proportion_dict, orient = 'index', columns = ['AirProportion'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Histogram thresholding on honeycomb data")
    parser.add_argument('--datasets', nargs='+', type=str, help='a list of datasets to process')
    args = parser.parse_args()

    if args.datasets != ['default']:
        datasets = args.datasets
    
    curr_dir_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    # Placeholder to store the proporition of air pixels in each dataset
    air_pixels_proportion_df_list = []
    
    # Call histogram_thresh() on the required datasets
    for dataset in datasets:
        if os.path.isdir(curr_dir_path + dataset_root + dataset):
            air_pixels_proportion_df_list.append(histogram_thresh(curr_dir_path + dataset_root + dataset))

            print('Creating histogram thresholding figure for ' + path)

            img = cv2.imread(curr_dir_path + dataset_root + dataset + '/tiff/0840.tiff', cv2.IMREAD_GRAYSCALE)
            mask, hist, thresh = get_hist_fg(img)
            
            # Create a pd.DataFrame containing all pixel intensities in the blurred image
            hist_fg_image_df = pd.DataFrame(cv2.GaussianBlur(img, (7, 7), 0).flatten(), columns=['PixelIntensity'])
            mask = hist_fg_image_df.PixelIntensity < thresh
            
            fig, axes = plt.subplots()
            
            # Plot the histogram, with bars appropriately colored
            ax = sns.histplot(data = hist_fg_image_df[hist_fg_image_df.PixelIntensity > 0], 
                              x = 'PixelIntensity', 
                              binwidth = 1, stat = 'proportion', 
                              hue=mask,
                              palette={True: 'gray', False: 'limegreen'},
                              element='bars', 
                              hue_order=[True, False],
                              legend=True, 
                              ax = axes)
            
            # Mark the threshold value used for foreground segmentation
            ax.axvline(x=thresh, c='lightcoral', ls='dashed', label = 'Threshold value')
            
            ax.legend(['Threshold value', 'Sample (Foreground)', 'Air (Background)'])
            
            ax.set_xlim(left = 0)
            ax.set_xlabel(r'Pixel Intensity ($\propto$ Radiodensity)')
            
            sns.despine()
            
            # Save the figure
            if not os.path.isdir(curr_dir_path + '../output/figures'):
                os.makedirs(curr_dir_path + '../output/figures')
            ax.figure.savefig(curr_dir_path + '../output/figures/histogram-segmentation-' + dataset + '.png', dpi=600)

            print('Plot for histogram thresholding for dataset ' + dataset + ' is saved at: ' + curr_dir_path + '../output/figures/histogram-segmentation-' + dataset + '.png')
    
    # Create a single dataset containing the proportion of air pixels for all datasets
    if air_pixels_proportion_df_list:
        air_pixels_proportion_df = pd.concat(air_pixels_proportion_df_list)
        air_pixels_proportion_df.to_csv(curr_dir_path + '../output/air_pixels_proportion_df.csv')
        print('air_pixels_proportion_df is saved at: ' + curr_dir_path + '../output/figures/histogram-segmentation-' + dataset + '.png')