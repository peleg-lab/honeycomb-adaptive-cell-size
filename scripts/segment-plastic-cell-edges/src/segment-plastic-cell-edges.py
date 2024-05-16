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

def segment_plastic_cell_edges(img, sigma, length):
    """
    This function finds and segments the plastic cell edges for an input image where the plastic base is removed.
    The image histogram is smoothed and its minima are found to identify the segmenatation threshold.

    Parameters:
    img (np.ndarray): A histogram thresholded image from the dataset.
    sigma (float): Sigma for the Gaussian convolution kernel.
    length (int): Length of Gaussian convolution kernel.
    
    Returns:
    mask (np.ndarray): Mask to use to filter out the plastic base.
    curve (np.ndarray): A 1-d numpy array containing the Gaussian smoothed histogram.
    threshold (int): Segmentation threshold for plastic cell edges.
    """
    norm = normalize(img)
    blurred_img = cv2.GaussianBlur(norm, (3, 3), 0)

    # Form the Gaussian smoothing kernel
    kernel = gaussian(length, sigma)
    kernel /= np.sum(kernel)

    # Calculate the histogram of the blurred image
    hist = cv2.calcHist([blurred_img[blurred_img > 0]],[0],None,[256],[0,256])
    hist = np.squeeze(hist)

    # Smooth the image histogram with a Gaussian kernel
    curve = np.convolve(hist, kernel, 'same')

    # Compute the minima of the smoothed image histogram and identify the threshold value
    minima = argrelextrema(curve, np.less)[0]
    threshold = minima[-1]
    # Check if minimum returned is a false positive
    if curve[threshold] > curve[threshold - 10] or curve[threshold] > curve[threshold + 10 if threshold + 10 < 256 else 255]:
        threshold = minima[-2]

    # Enhance the mask
    mask = np.where(norm > (threshold + 20), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    
    return mask, curve, threshold

def segment_plastic_cell_edges_o5x(tiff_image):
    """
    This function finds and segments the plastic cell edges for a o5x dataset image where the plastic base is removed.
    We write a separate function for o5x becasue the dataset is not as clean and requires special handling.

    Parameters:
    tiff_image (np.ndarray): A histogram thresholded TIFF image from the o5x dataset.
    
    Returns:
    np.ndarray: Mask to use to filter out the plastic cell edges.
    """
    img = cv2.GaussianBlur(tiff_image, (5, 5), 0)

    mask = np.logical_and(img > 21000, img < 22500) * 255
    mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > 2000:
            valid_contours.append(contour)

    plastic_segments = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    plastic_segments = cv2.drawContours(plastic_segments, valid_contours, -1, (255, 255, 255), -1)
    plastic_segments = cv2.cvtColor(plastic_segments, cv2.COLOR_BGR2GRAY)
    plastic_segments = cv2.morphologyEx(plastic_segments, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    
    return plastic_segments

# Driver code to segment 'teeth' of plastic starter
def segment_plastic_cell_edges_driver(path, sigma = 4.0, length = 15):
    """
    Driver function to segment the plastic cell edges for the dataset images

    Parameters:
    path (string): Path to the top level directory of the dataset.
    sigma (float): Sigma for the Gaussian convolution kernel used in segment_plastic_cell_edges().
    length (int): Length of Gaussian convolution kernel used in segment_plastic_cell_edges().

    Returns:
    No return value.
    """
    tiff_file_path =  path + '/plastic_base_masked/tiff'
    tiff_files = get_tiff_dict(tiff_file_path)

    curr_dir_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    plastic_segment_solution_path = curr_dir_path + f'../output/{path.split(os.path.sep)[-1]}/plastic_segmented/'

    # Create the output directory if it does not exist already
    if not os.path.isdir(plastic_segment_solution_path):
        os.makedirs(plastic_segment_solution_path)
        
    if not os.path.isdir(f'{plastic_segment_solution_path}/tiff/'):
        os.makedirs(f'{plastic_segment_solution_path}/tiff/')

    # Initialize video writer
    tiff_template = Image.open(tiff_files[tiff_file_path][0])
    temp_img = cv2.imread(tiff_files[tiff_file_path][0], cv2.IMREAD_GRAYSCALE)
    out = cv2.VideoWriter(plastic_segment_solution_path + path.split('/')[-1] + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (temp_img.shape[1], temp_img.shape[0]))

    print('Begin plastic cell edge segmentation of ' + path)
    for file in tqdm(sorted(tiff_files[tiff_file_path])):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        if path == 's_o5':
            plastic_segments_mask = segment_plastic_cell_edges_o5x(img)
        else:
            plastic_segments_mask, curve, threshold = segment_plastic_cell_edges(img, sigma, length)
            
        masked_image = img
        masked_image[plastic_segments_mask == 255] = 255
        
        tiff_image = cv2.imread(file, -1)
        tiff_masked = tiff_image
        tiff_masked[plastic_segments_mask == 255] = 0xffff
        
        write_tiff(tiff_template, f'{plastic_segment_solution_path}/tiff/{file.split(os.path.sep)[-1]}', tiff_masked)
        out.write(cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR))
    print('Finish plastic cell edge segmentation of ' + path)

    out.release()

if __name__ == '__main__':
    curr_dir_path = os.path.dirname(os.path.abspath(__file__)) + '/'

    parser = argparse.ArgumentParser(description="Histogram thresholding on honeycomb data")
    parser.add_argument('--datasets', nargs='+', type=str, help='a list of datasets to process')
    args = parser.parse_args()

    if args.datasets != ['default']:
        datasets = args.datasets
        
    #  Segment the plastic cell edges for the required datasets
    for dataset in datasets:
        if os.path.isdir(curr_dir_path + dataset_root + dataset + '/plastic_base_masked'):
            segment_plastic_cell_edges_driver(curr_dir_path + dataset_root + dataset)
    
            # Plot for plastic cell edge segmentation
            img = cv2.imread(curr_dir_path + dataset_root + dataset + '/plastic_base_masked/tiff/0840.tiff', cv2.IMREAD_GRAYSCALE)
            masked_image, curve, threshold = segment_plastic_cell_edges(img, 4, 15)
            
            # Form a DataFrame for the input image
            plastic_cell_edges_segmentation_image_df = pd.DataFrame(cv2.GaussianBlur(normalize(img), (3, 3), 0).flatten(), columns=['PixelIntensity'])
            df_mask = plastic_cell_edges_segmentation_image_df.PixelIntensity < threshold
            curve /= curve.sum()
            
            # Form a DataFrame for the curve
            plastic_cell_edges_segmentation_curve_df = pd.DataFrame(zip(range(len(curve)), curve), columns=['PixelIntensity', 'GaussianSmoothedHistogram'])
            plastic_cell_edges_segmentation_curve_df['Segment'] = 0
            plastic_cell_edges_segmentation_curve_df.loc[plastic_cell_edges_segmentation_curve_df.PixelIntensity >= threshold, 'Segment'] = 1

            print('Creating plastic cell edge segmentation figure for ' + dataset)
            fig, axes = plt.subplots()
            
            # Plot the histogram of pixel intensities in the input image, showing proportions
            ax = sns.histplot(data = plastic_cell_edges_segmentation_image_df[plastic_cell_edges_segmentation_image_df.PixelIntensity > 0], x = 'PixelIntensity', binwidth = 1, stat = 'proportion', hue=df_mask, palette={True: 'goldenrod', False: 'darkblue'},
                         element='bars', hue_order=[True, False], legend=True, ax = axes)
            ax.axvline(x=threshold, c='lightcoral', ls='dashed', label = 'Segmentation threshold')
            ax.legend(['Segmentation threshold', 'Plastic cell edges', 'Honeycomb'])#, title='Pixel intensity')
            
            # Plot the Gaussian smoothed histogram used to find the threshold value
            sns.lineplot(data=plastic_cell_edges_segmentation_curve_df, x='PixelIntensity', y='GaussianSmoothedHistogram', hue='Segment', palette={0: 'goldenrod', 1: 'darkblue'}, legend=None)
            
            ax.set_xlim(left = 0)
            
            ax.set_xlabel(r'Pixel Intensity ($\propto$ Radiodensity)')
            sns.despine()
            
            # Save the figure
            if not os.path.isdir(curr_dir_path + '../output/figures'):
                os.makedirs(curr_dir_path + '../output/figures')
            ax.figure.savefig(curr_dir_path + '../output/figures/plastic-cell-edge-segmentation-' + dataset + '.png', dpi=600)

            print('Figure for plastic cell edge segmentation for dataset ' + dataset + ' is saved at: ' + curr_dir_path + '../output/figures/plastic-cell-edge-segmentation-' + dataset + '.png')