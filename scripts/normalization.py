import os
import sys
import yaml
import argparse
import SimpleITK as sitk

from utils.utils_preprocessing import normalize 

# python scripts/run_processing_only --sub_id 127_S_0260 --img_dir ADNI_processed/processed_images --out_shape 128 128 128

#------------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    # SINGLE SUBJECT PROCESSING

    # directories
    parser.add_argument("--img_dir", type=str, default=os.path.join(os.getcwd(), "ADNI_processed", "processed_images"), help="output directory for preprocessed images")
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(os.getcwd(), "ADNI_processed", "ADNI_dataset"), help="path to dataset directory for model training")
    parser.add_argument("--brain", type=str, required=True, help="brain path")
    parser.add_argument("--mask", type=str, required=True, help="mask path")

    # subject ID
    parser.add_argument("--sub_id", type=str, required=True, help="subject ID, output folder name")

    # parameters
    parser.add_argument("--norm_range", type=float, nargs=2, default=[-1.0, 1.0], metavar="min, max", help="target intensity range for normalization")

    args = parser.parse_args()

    return args

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def normalizing(args):
    """
    Normalize brain and mask images using percentile clipping and rescale to a specified range.

    Parameters
    ----------
        args: Command line arguments 
            args.img_dir: str
                Directory containing the processed images
            args.dataset_dir: str
                Directory for final dataset storage for model training
            args.sub_id: str
                Subject ID
            args.norm_range: list of float
                Target intensity range for normalization    
    
    Returns
    -------
        None
    
    Note
    ----
        Intensity histograms of images before and after normalization are plotted in QC step
    """

    # subject ID = folder name
    #sub_id = os.path.basename(os.path.normpath(args.output_dir))
    sub_id = args.sub_id

    brain_path = args.brain
    mask_path = args.mask
    

    # SIMPLEITK IMAGE

    # Convert brain and mask to SimpleITK images :/
    brain_clipped = sitk.ReadImage(brain_path) # brain from FreeSurfer
    mask_clipped = sitk.ReadImage(mask_path) # mask from FreeSurfer

    # NORMALIZATION TO BRAIN RANGE 

    # Normalization: range defined in configuration
    brain_normalized = normalize(brain_clipped, mask_clipped, tuple(args.norm_range), is_label=False)
    mask_normalized =  normalize(mask_clipped, mask_clipped, tuple(args.norm_range), is_label=True)
    #guardare statistiche intensità: kl divergence tra immagini post gen

    # save normalized image
    brain_normalized_path = os.path.join(args.img_dir, sub_id, f'{sub_id}_brain.nii.gz') #FINAL BRAIN IMAGE
    mask_normalized_path = os.path.join(args.img_dir, sub_id, f'{sub_id}_mask.nii.gz') #FINAL MASK IMAGE
    sitk.WriteImage(brain_normalized, brain_normalized_path)
    sitk.WriteImage(mask_normalized, mask_normalized_path)

    print(f"Saved normalized brain image: {brain_normalized_path}")
    print(f"Saved normalized brain mask: {mask_normalized_path}")

    #return brain_normalized, mask_normalized


if __name__ == "__main__":
    args = parse_args()
    print("Running processing script for subject:", args.sub_id)
    normalizing(args)
