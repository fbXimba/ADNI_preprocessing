import os
import sys
import yaml
import argparse
import SimpleITK as sitk

from utils.utils_preprocessing import crop

# python scripts/run_processing_only --sub_id 127_S_0260 --img_dir ADNI_processed/processed_images --pad 2

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
    #parser.add_argument("--out_shape", type=int, nargs=3, default=[128, 128, 128], metavar="x,y,z", help="target output shape for resampling")
    parser.add_argument("--pad", type=int, default=2, help="padding around the brain for cropping")

    args = parser.parse_args()

    return args

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def cropping(args):

    """
    Crop brain and mask images with padding, images saved in processed_images/sub_id/

    Parameters
    ----------
        args: Command line arguments 
            args.img_dir: str
                Directory containing the processed images
            args.sub_id: str
                Subject ID
            args.pad: int
                Padding around the brain for cropping

    Returns
    -------
        None

    """

    sub_id = args.sub_id # subject ID = folder name
    
    # load FreeSurfer brain and mask paths
    #brain_path = os.path.join(args.img_dir, sub_id, f'{sub_id}_brain_fs.nii.gz')
    #mask_path = os.path.join(args.img_dir, sub_id, f'{sub_id}_mask_fs.nii.gz')
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

    # CROPPING WITH PADDING to cubic bounding box

    # Crop images with padding
    brain_cropped, mask_cropped, box = crop(brain_sitk, mask_sitk, pad=args.pad)

    # save cropped images
    brain_cropped_path = os.path.join(args.img_dir, sub_id, f'{sub_id}_brain_cropped.nii.gz')
    mask_cropped_path  = os.path.join(args.img_dir, sub_id, f'{sub_id}_mask_cropped.nii.gz') 
    sitk.WriteImage(brain_cropped, brain_cropped_path)
    sitk.WriteImage(mask_cropped, mask_cropped_path)

    print(f"Saved and cropped brain mask: {mask_cropped_path}")
    print(f"Saved and cropped brain image: {brain_cropped_path}")


    #return brain_cropped, mask_cropped

if __name__ == "__main__":
    args = parse_args()
    print("Running processing script for subject:", args.sub_id)
    cropping(args)
