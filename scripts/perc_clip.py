import os
import sys
import yaml
import argparse
import SimpleITK as sitk

from utils.utils_preprocessing import clipping 

# python scripts/perc_clip.py --sub_id 127_S_0260 --brain <brain_path> --mask <mask_path>

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
    parser.add_argument("--norm_perc", type=float, nargs=2, default=[0.1, 99.9], metavar="min, max", help="percentiles for intensity clipping")

    args = parser.parse_args()

    return args

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def clipping_step(args):
    """
    Clip brain intensities using percentile clipping.

    Parameters
    ----------
        args: Command line arguments 
            args.img_dir: str
                Directory containing the processed images
            args.sub_id: str
                Subject ID
            args.norm_perc: list of float
                Percentiles for intensity clipping
    
    Returns
    -------
        None
    """

    # subject ID = folder name
    sub_id = args.sub_id

    brain_path = args.brain
    mask_path = args.mask

    # SIMPLEITK IMAGE

    # Convert brain and mask to SimpleITK images :/
    brain_fixed = sitk.ReadImage(brain_path) # brain from FreeSurfer
    mask_fixed = sitk.ReadImage(mask_path) # mask from FreeSurfer

    # CLIPPING: perc defined in configuration
    brain_clipped = clipping(brain_fixed, mask_fixed, tuple(args.norm_perc), is_label=False)
    mask_clipped = clipping(mask_fixed, mask_fixed, (0, 100), is_label=True)

    # save clipped image
    brain_clipped_path = os.path.join(args.img_dir, sub_id, f'{sub_id}_brain_clipped.nii.gz')
    mask_clipped_path = os.path.join(args.img_dir, sub_id, f'{sub_id}_mask_clipped.nii.gz')
    sitk.WriteImage(brain_clipped, brain_clipped_path)
    sitk.WriteImage(mask_clipped, mask_clipped_path)

    print(f"Saved clipped brain image: {brain_clipped_path}")
    print(f"Saved clipped brain mask: {mask_clipped_path}")


if __name__ == "__main__":
    args = parse_args()
    print("Running clipping script for subject:", args.sub_id)
    clipping_step(args)
