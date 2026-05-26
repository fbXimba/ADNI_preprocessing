
"""
utils_preprocessing.py
======================

Preprocessing functions for MRI images and masks.

This module provides utility functions for preprocessing MRI images and corresponding masks, including:

- Resampling images and masks to a reference geometry
- Cropping images and masks to a bounding box
- Resizing images by resampling to a target shape
- Normalizing MRI intensities with percentile-based clipping

All functions use SimpleITK and NumPy for image manipulation and are designed for use in neuroimaging pipelines.

Examples
--------
>>> img_out = check_to_reference(img, ref, is_label=True)
>>> img_cropped, mask_cropped, box = crop(img, mask, pad=5)
>>> img_resized = resize_resample(img, (128, 128, 128), is_label=False)
>>> img_norm = normalize(img, mask, (0.5, 99.5), (-1, 1))

"""



# libraries
import SimpleITK as sitk # for image processing
import numpy as np
import argparse # for argument parsing

#------------------------------------------------------------------------------------------

#FUNCTIONS+

INTERPOLATORS = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline3
}

def yn_to_bool(value):
    """
    Convert 'y'/'n' string to boolean True/False 
    Used due to argparse limitations with boolean arguments

    Parameters
    ----------
        value: str
            input string ('y' or 'n')   
    Returns
    -------
        boolean
            True if 'y', False if 'n'
    Raises
    ------
        argparse.ArgumentTypeError
            if input is not 'y' or 'n'
    Notes
    -----
        - case insensitive
        - useful for command line argument parsing   
    """

    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('y', 'yes'):
        return True
    elif value in ('n', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError("Expected 'y' or 'n'.")


# resample image (mask) to match reference image (brain) geometry
def check_to_reference(img, ref, is_label):

    """
    Resample mask to match brain in size, spacing, origin, and direction

    Necessary for preprocessing steps and histograms creation of original images

    Parameters
    ----------
        img: SimpleITK image 
            image to resample (mask)
        ref: SimpleITK image
            reference image to copy geometry from (brain mri)
        is_label: boolean
            whether the image is a label image (True = mask) or not (False = mri)

    Returns
    -------
        img_out: SimpleITK image
            resampled image to match reference image geometry (mask resampled to brain)

    Notes
    -----
        - using nearest neighbor interpolation for label images = masks --> preserves boolean values and avoids artifacts
        - using bspline interpolation for non-label images = mri --> preserves intensity values and avoids artifacts

    """

    # Check if mask and image are in the same space
    same = (img.GetSize() == ref.GetSize()
        and np.allclose(img.GetSpacing(), ref.GetSpacing())
        and np.allclose(img.GetOrigin(), ref.GetOrigin())
        and np.allclose(img.GetDirection(), ref.GetDirection())
    )
    # if NOT resample mask to match brain: mask still good!
    if not same:
        print("Resampling freesurfer mask to match adni mri")
        
        interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline3

        img_out = sitk.Resample(
            img,
            ref,                # reference image to copy geometry from
            sitk.Transform(),   # identity transform
            interpolator,
            0,                  # background value
            img.GetPixelID()
        )

    else:
        img_out = img

    return img_out

def new_coord(min_val, max_val, measure, pad, img_dim):
    """
    Adjusts coordinates of a bounding box to include padding, remain cubic and ensure it fits within image dimensions

    parameters
    ----------
        min_val: int
            minimum coordinate of the bounding box (x_min, y_min, z_min)
        max_val: int
            maximum coordinate of the bounding box (x_max, y_max, z_max)
        measure: int
            size of the bounding box along the largest dimension
        pad: int
            padding to add around the bounding box
        img_dim: int
            size of the image along the corresponding dimension

    returns
    -------
        min_val: int
            adjusted minimum coordinate for the bounding box
        max_val: int                
            adjusted maximum coordinate for the bounding box
    """

    min_val = max(min_val - pad, 0)
    max_val = min(min_val + measure + pad, img_dim)

    if max_val - min_val < measure + 2*pad:

        if min_val == 0:
            max_val = min(min_val + measure + 2*pad, img_dim)

        else:
            min_val = max(max_val - measure - 2*pad, 0)

    return min_val, max_val

# crops mask and brain to bounding box
def crop(img_sitk, mask_sitk, pad):

    """
    Crop the image and mask to a bounding box around the brain mask, with optional padding

    Parameters
    ----------
        img_sitk: SimpleITK image
            mri image
        mask_sitk: SimpleITK image
            boolean brain mask
        pad: int
            padding around the brain in voxels

    Returns
    -------
        img_cropped: SimpleITK image
            cropped MRI to bounding box
        mask_cropped: SimpleITK image
            cropped mask to bounding box
        box: tuple
            bounding box in voxel coordinates (xmin, xmax, ymin, ymax, zmin, zmax)

    Notes
    -----
        - checks if the mask and image are in the same space: otherwise resample the mask
        - padding is added to each side of the bounding box, clipped to image bounds
        - box can be useful later for reversing crop or for reference

    """

    # Convert mask to boolean numpy array
    mask_array = sitk.GetArrayFromImage(mask_sitk).astype(bool)  # [z, y, x]
    coords = np.argwhere(mask_array) #non-zero coord
    if coords.size == 0:
        raise ValueError("Mask is empty") #check
    
    # Coordinates for cropping
    z_min, y_min, x_min = coords.min(axis=0) # min coords
    z_max, y_max, x_max = coords.max(axis=0) + 1  # max coords : +1 because slicing is exclusive

    # Image size in SimpleITK order (x, y, z)
    img_size = img_sitk.GetSize() 

    # max measurement to keep cubi box
    measure = max(x_max - x_min, y_max - y_min, z_max - z_min) 

    if measure + pad > max(img_size):
        print("Warning: padding too large for image size, reducing to fit")
        pad = (max(img_size) - measure) // 2

    x_min, x_max = new_coord(x_min, x_max, measure, pad, img_size[0])
    y_min, y_max = new_coord(y_min, y_max, measure, pad, img_size[1])
    z_min, z_max = new_coord(z_min, z_max, measure, pad, img_size[2])

    # Keep bounding box if useful later ?
    box = (x_min, x_max, y_min, y_max, z_min, z_max)

    # This way because SimpleITK expects [x, y, z]
    index = [int(x_min), int(y_min), int(z_min)]
    size = [int(x_max - x_min), int(y_max - y_min), int(z_max - z_min)] # should be the same for all dimensions

    # Crop
    img_cropped = sitk.RegionOfInterest(img_sitk, size=size, index=index)
    mask_cropped = sitk.RegionOfInterest(mask_sitk, size=size, index=index)

    return img_cropped, mask_cropped, box # see box no cubico + resampla isotropico --> spostato in basso a sinistra


# Resize by resampling to desired shape
def resize_resample(img, target_size, is_label):

    """
    Resample to resize image to a target size 
    (without isotropic spacing)


    Parameters
    ---------- 
        img: SimpleITK image 
            image to resample
        target_size: tuple
            target shape/size (tuple in parameters)
        is_label: boolean
            whether the image is a label image (True = mask) or not (False = mri)

    Returns
    -------
        img_out: SimpleITK image
            resampled SimpleITK image  of desired shape without isotropic spacing

    Notes
    -----
        - if img.GetSize() != target_size to avoid unecessary actions
        - using linear interpolation for non-label images = mri --> preserves intensity values and avoids artifacts
        - using nearest neighbor for label images = masks --> preserves boolean values and avoids artifacts

    """

    if img.GetSize() != target_size:
        original_size = img.GetSize()
        original_spacing = img.GetSpacing()

        # new spacing = original spacing * (original size / target size) # TODO: fisso isotropico per crop
        new_spacing = [
            original_spacing[0] * (original_size[0] / target_size[0]),
            original_spacing[1] * (original_size[1] / target_size[1]),
            original_spacing[2] * (original_size[2] / target_size[2]),
        ]

        # Resampler configuration: create (1) + set (others-1) + execute (last)
        resampler = sitk.ResampleImageFilter()  #create
        resampler.SetOutputSpacing(new_spacing) #output spacing
        resampler.SetSize(target_size) #output size
        resampler.SetOutputDirection(img.GetDirection()) #output direction
        resampler.SetOutputOrigin(img.GetOrigin()) #output origin
        resampler.SetTransform(sitk.Transform()) #transform
        resampler.SetDefaultPixelValue(0) #default pixel value
        resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear) #interpolator #spline mri
        img_out = resampler.Execute(img) #execute
    else:
        # Image is already the target size, return as-is
        img_out = img

    return img_out


# normalize mri intensities 
def clipping(img, mask, perc, is_label):

    """
    Clip mri intensities with percentiles

    Parameters
    ----------
        img: SimpleITK image
            image to clip
        mask: SimpleITK image
            brain mask used to compute clipping and min/max for mri
        perc: tuple
            percentile for clipping
        is_label: boolean
            whether the image is a label image (True = mask) or not (False = mri)

    Returns
    -------
        img_out: SimpleITK image
            clipped image with original metadata

    """

    if is_label: # only for mri
        # Convert to numpy array
        img_array = sitk.GetArrayFromImage(img).astype(np.float32)  #

    else: # for mri
        # Convert to numpy array
        img_array = sitk.GetArrayFromImage(img).astype(np.float32)  # [z, y, x]
        mask_array = sitk.GetArrayFromImage(mask).astype(bool) # non-zero voxels = brain

        # brain mask to look at intensities distribution to remove background
        brain= img_array[mask_array]

        # percentile clipping to remove outliers
        p_low, p_high = np.percentile(brain, perc)
        img_array = np.clip(img_array, p_low, p_high)

    # Convert back to SimpleITK image
    img_out = sitk.GetImageFromArray(img_array) # back to SimpleITK from array
    img_out.CopyInformation(img)  # copy original image metadata!!

    return img_out


# normalize mri intensities 
def normalize(img, mask, range, is_label):

    """
    Normalize mri intensities to a target range

    Parameters
    ----------
        img: SimpleITK image
            image to normalize
        perc: tuple
            percentile for clipping
        range: tuple
            target intensity range
        is_label: boolean
            whether the image is a label image (True = mask) or not (False = mri)

    Returns
    -------
        img_out: SimpleITK image
            normalized SimpleITK image with original metadata

    """

    if is_label: # only for mri
        # Convert to numpy array
        img_array = sitk.GetArrayFromImage(img).astype(np.float32)  #

        img_min, img_max = img_array.min(), img_array.max() # use all voxels

    else: # for mri
        # Convert to numpy array
        img_array = sitk.GetArrayFromImage(img).astype(np.float32)  # [z, y, x]
        mask_array = sitk.GetArrayFromImage(mask).astype(bool) # non-zero voxels = brain

        # Normalize to target range (-1,1)
        img_min, img_max = img_array[mask_array].min(), img_array[mask_array].max() # use brain voxels only
        
    img_array = ((img_array - img_min) / (img_max - img_min)) * (range[1] - range[0]) + range[0]  # scale to [0, 1] then scale to range

    # Convert back to SimpleITK image
    img_out = sitk.GetImageFromArray(img_array) # back to SimpleITK from array
    img_out.CopyInformation(img)  # copy original image metadata!!
     
    return img_out
