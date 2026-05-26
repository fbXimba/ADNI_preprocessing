
"""
utils_QC.py
===========

Quality Control functions for images and histograms.

This module provides utility functions for generating quality control (QC) images and histograms for MRI data and masks, including:

- Creating QC figures to visually compare processed and original images
- Plotting intensity histograms for brain images with masking and thresholding
- Saving QC outputs for later inspection

All functions use SimpleITK, NumPy, and Matplotlib for image manipulation and visualization, and are designed for use in neuroimaging pipelines.

Examples
--------
>>> qc_image(processed, original, sub_id, qc_dir, dir_home)
>>> qc_hist(img, img_orig, mask, mask_orig, sub_id, output_dir, dir_home)

"""



# libraries
import os
#import subprocess # for running FreeSurfer commands to create masks
#import glob # to get all subject paths
import SimpleITK as sitk # for image processing
#import nibabel as nib # to read .mgz files
import numpy as np
import matplotlib.pyplot as plt

# functions
from utils.utils_preprocessing import check_to_reference



# FUNCTIONS FOR QUALITY CONTROL IMAGES AND HISTOGRAMS OF PREPROCESSED IMAGES
#------------------------------------------------------------------------------------------


def qc_image(processed, original, sub_id, slices_dir, dir_home):

    """
    Creates QC figure to check whether processed image is acceptable against original image 
    Saves it in slices_dir with subject id

    1st row processed: sagittal, coronal, axial
    2nd row original: axial, coronal, sagittal

    Parameters
    ----------
        processed: SimpleITK image
            processed image to check
        original: SimpleITK image
            original image to compare against
        sub_id: str
            subject ID
        qc_dir: str
            directory to save QC figure
        dir_home: str
            home directory to return after saving figure

    Returns
    -------
        None
            Creates and saves QC figure

    Notes
    -----
        - plots in different order the 3 views (slices) of processed and original images due to FreeSurfer convention
        - middle slices for each view
        - not the same slices for processed and original, but should be close enough
        - aspect ratio not preserved

    """

    # aspect ratio for matplotlib not implemented

    # array from SimpleITK image to numpy for visualization withshape [z,y,x]
    processed_array = sitk.GetArrayFromImage(processed) 
    original_array = sitk.GetArrayFromImage(original)

    # middle slices for axial, coronal, sagittal views
    mid_slices = [processed_array.shape[0]//2, processed_array.shape[1]//2, processed_array.shape[2]//2] 
    #original_slices = [original_array.shape[0]//2, original_array.shape[1]//2, original_array.shape[2]//2]

    # move to slices directory
    os.chdir(slices_dir) 

    # create figure with subplots for each view
    # sagittal, coronal, axial for original processed outputs and original images
    # coronal, axial, sagittal for freesurfer processed outputs
    fig, axes = plt.subplots(2, 3, figsize=(10, 3))    
    for i in [[processed_array, "p",["Coronal", "Axial", "Sagittal"]], [original_array, "o", ["Sagittal", "Coronal", "Axial"]]]:
        ax1,ax2,ax3= axes[0] if i[0] is processed_array else axes[1]
        ax1.imshow(i[0][mid_slices[0], :, :], cmap='gray')
        ax1.set_title(f'{i[2][0]} {i[1]}') 
        ax1.axis('off')
        ax2.imshow(i[0][:, mid_slices[1], :], cmap='gray')
        ax2.set_title(f'{i[2][1]} {i[1]}')
        ax2.axis('off')
        ax3.imshow(i[0][:, :, mid_slices[2]], cmap='gray')
        ax3.set_title(f'{i[2][2]} {i[1]}')
        ax3.axis('off')
    plt.tight_layout() # tight layout for better spacing
    plt.savefig(f"{sub_id}_qc.png", dpi=120)
    plt.close(fig)
    print(f"Saved QC image: {sub_id}_qc.png")

    # go back to home directory
    os.chdir(dir_home) 

    return None


def plot_hist(ax, data, bins, color, title, count_threshold):

    """
    Creates an histogram of image intensities for QC histograms

    Only showing bins with counts above a threshold with mask

    Parameters
    ----------
        ax: matplotlib axis 
            axis to plot on (from subplots)
        data: np array
            already flattened data array of image intensities (brain only) for histogram
        bins: int
            number of bins for histogram (param qc_hist)
        color: str
            color for histogram bars
        title: str
            title for histogram
        count_threshold: int
            minimum count to display a bin (param qc_hist)

    Returns
    -------
        Plots threshold histogram on given axis

    Notes
    -----
        - mask BINS below threshold (necessarily on bins and not single values!! would affect overall distribution)
        - using bar plot to be able to mask bins below threshold
        - histogram with counts and bin edges
        - facecolor="none" for steplikeish

    """

    # compute histogram for counts and bin edges
    counts, bin_edges = np.histogram(data, bins=bins)

    # mask bins below threshold (necessarily on bins and not single values!!)
    mask = counts >= count_threshold

    # plotting using left edges and heights
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.bar(bin_centers[mask], counts[mask], width=(bin_edges[1]-bin_edges[0]), color=color, edgecolor=color, alpha=0.7, lw=1.5) 
    ax.set_title(title)
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Counts')
    ax.grid(axis='both',alpha=0.3)

    return None


def qc_hist(img,img_orig, mask, mask_orig, sub_id, output_dir, dir_home): #count threshold

    """
    Create histogram of image intensities for QC, only showing bins with counts above a threshold

    Save figure in output_dir with subject id

    Parameters
    ----------
        img: SimpleITK image
            FreeSurfer processed brain image
        img_orig: SimpleITK image
            original masked brain image
        mask: SimpleITK image
            FreeSurfer processed brain mask image
        mask_orig: SimpleITK image
            original FreeSurfer brain mask image
        sub_id: str
            subject ID
        output_dir: str
            directory to save histogram figure
        dir_home: str
            home directory to return after saving
        #count_threshold: int
        #   minimum count to display a bin

    Returns
    -------
        None
            Saves QC histograms in output_dir with subject id

    """
    #NOTE: qualcosa non torna nei counts X(

    # array from SimpleITK image to numpy for visualization
    img_array = sitk.GetArrayFromImage(img) 
    img_orig_array = sitk.GetArrayFromImage(img_orig)
    mask_array = sitk.GetArrayFromImage(mask).astype(bool) # non-zero voxels = brain

    # adapt mask for original image
    mask_orig = check_to_reference(mask, img_orig, is_label=True)
    mask_orig_array = sitk.GetArrayFromImage(mask_orig).astype(bool) # non-zero voxels = brain

    # brain only for histogram
    brain_array = img_array[mask_array]
    brain_orig_array = img_orig_array[mask_orig_array]

    # move to output directory
    os.chdir(output_dir) 

    # create histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))
    bins = 50
    count_threshold = 100  # minimum count to display a bin

    # FS processed
    plot_hist(axes[0], brain_array.flatten(), bins, 'blue', f'FS processed {sub_id}', 1)

    # Original masked
    plot_hist(axes[1], brain_orig_array.flatten(), (bins+10), 'red', f'M Original {sub_id}', count_threshold)

    #plt.tight_layout()
    plt.savefig(f"{sub_id}_hist.png", dpi=120)
    plt.close()

    print(f"Saved histogram image: {sub_id}_hist.png")

    # go back to home directory
    os.chdir(dir_home)

    return None