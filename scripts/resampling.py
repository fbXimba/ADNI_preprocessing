import os
import sys
import yaml
import argparse
import SimpleITK as sitk

from utils.utils_preprocessing import resize_resample # crop

# python scripts/run_processing_only --sub_id 127_S_0260 --img_dir ADNI_processed/processed_images --out_shape 128 128 128

#------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    # SINGLE SUBJECT PROCESSING

    # directories
    parser.add_argument("--img_dir", type=str, default=os.path.join(os.getcwd(), "ADNI_processed", "processed_images"), help="output directory for preprocessed images")
    parser.add_argument("--brain", type=str, required=True, help="brain path")
    parser.add_argument("--mask", type=str, required=True, help="mask path")

    # subject ID
    parser.add_argument("--sub_id", type=str, required=True, help="subject ID, output folder name")

    # parameters
    parser.add_argument("--out_shape", type=int, nargs=3, default=[128, 128, 128], metavar="x,y,z", help="target output shape for resampling")
    #parser.add_argument("--pad", type=int, default=2, help="padding around the brain for cropping")

    args = parser.parse_args()

    return args

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def resampling(args):

    """
    Resample brain and mask images to a fixed output shape, images saved in processed_images/sub_id/

    Parameters
    ----------
        args: Command line arguments 
            args.img_dir: str
                Directory containing the processed images
            args.sub_id: str
                Subject ID
            args.out_shape: list of int
                Target output shape for resampling

    Returns
    -------
        None

    Note
    ----
        cropping already performed, resampling to fixed output shape follows
    """

    sub_id = args.sub_id # subject ID = folder name
    
    # load FreeSurfer brain and mask paths
    #brain_path = os.path.join(args.img_dir, sub_id, f'{sub_id}_brain_cropped.nii.gz')
    #mask_path = os.path.join(args.img_dir, sub_id, f'{sub_id}_mask_cropped.nii.gz')
    brain_path = args.brain
    mask_path = args.mask
    
    # check files exist
    if not os.path.exists(brain_path):
        print(f"ERROR: Brain file not found: {brain_path}")
        sys.exit(1)
    if not os.path.exists(mask_path):
        print(f"ERROR: Mask file not found: {mask_path}")
        sys.exit(1)
    
    # SIMPLEITK IMAGE from FreeSurfer outputs
    brain_sitk = sitk.ReadImage(brain_path)
    mask_sitk = sitk.ReadImage(mask_path)
    #brain_og_sitk = sitk.ReadImage(os.path.join(args.img_dir, sub_id, f'{sub_id}_brain_og.nii.gz')) # original brain with skull stripping

    print(f"brain mask size: {mask_sitk.GetSize()}, spacing: {mask_sitk.GetSpacing()}, origin: {mask_sitk.GetOrigin()}")
    print(f"brain image size: {brain_sitk.GetSize()}, spacing: {brain_sitk.GetSpacing()}, origin: {brain_sitk.GetOrigin()}")
    #print(f"original brain image size: {brain_og_sitk.GetSize()}, spacing: {brain_og_sitk.GetSpacing()}, origin: {brain_og_sitk.GetOrigin()}")

    # RESAMPLING-RESIZING TO FIXED CUBIC SHAPE # evaluate maxpool without anisotropic crop

    # Resize to fixed output shape
    brain_fixed = resize_resample(brain_sitk, tuple(args.out_shape), is_label=False)
    mask_fixed = resize_resample(mask_sitk, tuple(args.out_shape), is_label=True)

    # save resampled images
    brain_fixed_path = os.path.join(args.img_dir, sub_id, f'{sub_id}_brain_resampled.nii.gz') # add _{tuple(args.out_shape)}[0] ?
    mask_fixed_path  = os.path.join(args.img_dir, sub_id, f'{sub_id}_mask_resampled.nii.gz') 
    sitk.WriteImage(brain_fixed, brain_fixed_path)
    sitk.WriteImage(mask_fixed, mask_fixed_path)

    print(f"Saved and resampled brain mask: {mask_fixed_path}")
    print(f"Saved and resampled brain image: {brain_fixed_path}")


    #return brain_fixed, mask_fixed


if __name__ == "__main__":
    args = parse_args()
    print("Running processing script for subject:", args.sub_id)
    resampling(args)
