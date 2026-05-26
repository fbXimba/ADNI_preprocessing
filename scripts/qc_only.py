import os
import sys
import SimpleITK as sitk
import glob
import yaml
import argparse

from utils.utils_QC import qc_image, qc_hist

# python -m scripts.run_qc_only --subj_dir ADNI/127_S_0260 --output_dir ADNI_processed --qc_dir ADNI_processed/brain_qc

#------------------------------------------------------------------------------------------

# CONFIGURATION

def parse_args():
    parser = argparse.ArgumentParser()
    # default from config file, options to override from command line
    # SINGLE SUBJECT PROCESSING

    # directories
    parser.add_argument("--sub_id", type=str, required=True, help="Subject ID")
    parser.add_argument("--nii_file", type=str, required=True, help="input NIfTI file of original image")
    #parser.add_argument("--img_dir", type=str, default=os.path.join(os.getcwd(), "ADNI_processed", "processed_images"), help="output directory for preprocessed images")
    parser.add_argument("--qc_dir", type=str, default=os.path.join(os.getcwd(), "ADNI_processed", "brain_qc"), help="output directory for qc images")
    parser.add_argument("--brain", type=str, required=True, help="brain path")
    parser.add_argument("--mask", type=str, required=True, help="mask path")
    parser.add_argument("--brain_og", type=str, required=True, help="brain path")
    parser.add_argument("--mask_fs", type=str, required=True, help="mask path")


    ## QC
    #parser.add_argument("--qc_enable", type=yn_to_bool, default=('y' if qc_enable=='y' else 'n'), help="enable quality control step (y/n)")

    args = parser.parse_args()

    return args

# initialize arguments from command line)
if __name__ == "__main__":
    args = parse_args()
    print("Running QC script")

#------------------------------------------------------------------------------------------

def QC(args):

    # subject ID = folder name
    sub_id = args.sub_id
    sub_path = args.nii_file
    #sub_id = os.path.basename(os.path.normpath(args.subj_dir))

    #nii_files = glob.glob(os.path.join(args.subj_dir, "**", "*.nii.gz"), recursive=True)
    #sub_path = nii_files[0]  # assuming there's at least one .nii file

    brain_path = args.brain
    mask_path = args.mask
    brain_og_path = args.brain_og
    mask_fs_path = args.mask_fs

    dir_home = os.getcwd()
    hist_dir = os.path.join(args.qc_dir, 'hist')
    slices_dir = os.path.join(args.qc_dir, 'slices')

    os.makedirs(args.qc_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)
    os.makedirs(slices_dir, exist_ok=True)

    # QUALITY CONTROL SLICES IMAGE

    brain_normalized = sitk.ReadImage(brain_path)
    adni_sitk = sitk.ReadImage(sub_path)
    brain_og_sitk = sitk.ReadImage(brain_og_path)
    mask_fixed = sitk.ReadImage(mask_path)
    # load original FreeSurfer mask as SimpleITK image (was mistakenly left as path string)
    mask_sitk = sitk.ReadImage(mask_fs_path)

    # qc image saved in slices_dir with subject id then back to dir_home
    qc_image(brain_normalized, adni_sitk, sub_id, slices_dir, dir_home)

    # QUALITY CONTROL HISTOGRAMS OF INTENSITIES

    # qc histograms saved in hist_dir with subject id then back to dir_home
    qc_hist(brain_normalized, brain_og_sitk, mask_fixed, mask_sitk, sub_id, hist_dir, dir_home) 


if __name__ == "__main__":
    QC(args)
