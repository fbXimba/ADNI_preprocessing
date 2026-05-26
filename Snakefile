import os
import pandas as pd
from snakemake.io import expand
import glob

# snakemake --use-conda --cores 75 --rerun-incomplete --latency-wait 900 --keep-going -p 

#try with export OMP_NUM_THREADS=1 for slow freesurfer problem
"""
NOTE: running scripts with conda run -n env-name python script.py ... atm, works with .yaml env file also but not with the name of the env
    conda:
        "envs/env-prep.yaml" # this way it creates the enviroment, works: should maybe check if already existing
       # "/home/francesca/miniconda3/envs/env-prep"  and "env.prep" DO NOT WORK :(  
"""
#---------------------------------------------------------------------------------------------------------------------------------

# Load configuration
configfile: "config_pre.yaml"

# Project root
PROJECT_ROOT = os.getcwd() #os.path.dirname(os.path.abspath(__file__)) # sarebbe meglio ma non va qc :(

# Directories from config
input_dir = config["directories"]["input_dir"]
output_dir = config["directories"]["output_dir"]
processed_dir = config["directories"]["img_dir"]
fs_dir = config["directories"]["fs_dir"]
dataset_dir = config["directories"]["dataset_dir"]

# Subject range
subj_range = config["snakemake"]["subj_range"]

# preprocessing parameters
min_mask_voxels = config["preprocessing"]["min_mask_voxels"]
pad = config["preprocessing"]["pad"]
out_shape = config["preprocessing"]["out_shape"]
norm_perc = config["preprocessing"]["norm_perc"]
norm_range = config["preprocessing"]["norm_range"]


csv_file = config["snakemake"]["csv_file"]
df = pd.read_csv(csv_file)

# Functions

def get_subject_file(wildcards):
   """Get the NIfTI file for a given subject from the CSV, returns ONE file path for each subject"""
   return subject_files[wildcards.sub_id]

# Exclude failed subjects
def has_failure(sub_id):
    return (os.path.exists(os.path.join(fs_dir, sub_id, 'FS_FAILED')) and # changed output_dir to fs_dir
            os.path.exists(os.path.join(processed_dir, sub_id, 'PROC_FAILED')))

#---------------------------------------------------------------------------------------------------------------------------------

# Create necessary directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fs_dir, exist_ok=True)

# Map subjects id - files from CSV file
subject_files = {}
for _, row in df.iterrows(): # iterate over DataFrame rows
    subject_id = row['Subject']
    image_id = row['Image Data ID']
    
    # Use glob to find the actual file matching this pattern
    # Pattern matches: ADNI/127_S_0260/MPR__GradWarp__B1_Correction__N3__Scaled/YYYY-MM-DD_HH_MM_SS.S/I86120/ADNI_*_I86120.nii
    # ADNI = input_dir/127_S_0260 = [subject_id]/[scan_type]/[timestamp]/I86120 = [image_id]/*_[image_id].nii = *_[image_id].nii
    search_pattern = os.path.join(input_dir, subject_id, "*", "*", image_id, f"*_{image_id}.nii")
    
    # Find all matching files
    matching_files = glob.glob(search_pattern)
    
    if matching_files:
        # Store only the first matching file (should be exactly one per Image Data ID)
        subject_files[subject_id] = matching_files[0]
    else:
        print(f"WARNING: No files found for Subject {subject_id}, Image ID {image_id}")
        print(f"  Search pattern was: {search_pattern}")
        
# Get selected subjects based on range
start, end = subj_range
SUBJECTS = sorted(list(subject_files.keys()))[start:end]

SELECTED_SUBJECTS = [s for s in SUBJECTS if not has_failure(s)]
#print("SELECTED_SUBJECTS:", SELECTED_SUBJECTS)

# Targets
DATASET_HEADS = expand(os.path.join(dataset_dir, "image", "{sub_id}_brain.nii.gz"), sub_id=SELECTED_SUBJECTS)
DATASET_MASKS = expand(os.path.join(dataset_dir, "mask", "{sub_id}_mask.nii.gz"), sub_id=SELECTED_SUBJECTS)

#---------------------------------------------------------------------------------------------------------------------------------
# Rules

rule all:
    input:
        DATASET_HEADS,
        DATASET_MASKS

rule freesurfer_autorecon1:
    input:
        nii_file=get_subject_file 
        # alternative : lambda wildcards: subject_files[wildcards.sub_id] 
    output:
        head_fs=os.path.join(fs_dir, "{sub_id}", "mri", "T1.mgz")
    params:
        outdir=output_dir,
        sub_id="{sub_id}",
        fs_dir=fs_dir
    #threads: 1
    #resources:
    #    mem_mb=8000
    log:
        "logs/{sub_id}.autorecon1.log"
    shell:
        r"""
        set -eo pipefail
        mkdir -p logs {params.fs_dir}/{wildcards.sub_id}
        export OMP_NUM_THREADS=1
        export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
        rm -rf {params.fs_dir}/{wildcards.sub_id}  
        recon-all -i {input.nii_file} -s {params.sub_id} -sd {params.fs_dir} -autorecon1 -no-isrunning >> {log} 2>&1
        """

rule freesurfer_mri_convert:
    input:
        T1=os.path.join(fs_dir, "{sub_id}", "mri", "T1.mgz"),
    output:
        head_fs=os.path.join(processed_dir, "{sub_id}", "{sub_id}_head_fs.nii.gz")
    params:
        img_dir=processed_dir
    log:
        "logs/{sub_id}.mgz_to_nii.log"
    shell:
        r"""
        set -eo pipefail
        export OMP_NUM_THREADS=1
        export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
        mkdir -p logs {params.img_dir}/{wildcards.sub_id}
        mri_convert {input.T1} {output.head_fs} >> {log} 2>&1
        """

rule head_mask:
    input:
        head_fs=os.path.join(processed_dir, "{sub_id}", "{sub_id}_head_fs.nii.gz")
    output:
        mask_head=os.path.join(processed_dir, "{sub_id}", "{sub_id}_mask_itk.nii.gz")
    params:
        img_dir=processed_dir
    log:
        "logs/{sub_id}.mask_itk.log"
    shell:
        r"""
        set -euo pipefail
        export PYTHONPATH="${{PYTHONPATH:-}}:{PROJECT_ROOT}"
        mkdir -p logs {params.img_dir}/{wildcards.sub_id}
        conda run -n env-prep python scripts/head_extraction_from_mri.py \
            --input {input.head_fs} \
            --output {output.mask_head} \
            >> {log} 2>&1
        """

rule crop:
    input:
        head_fs=os.path.join(processed_dir, "{sub_id}", "{sub_id}_head_fs.nii.gz"),
        mask_head=os.path.join(processed_dir, "{sub_id}", "{sub_id}_mask_itk.nii.gz")
    output:
        brain_cr=os.path.join(processed_dir, "{sub_id}", "{sub_id}_brain_cropped.nii.gz"),
        mask_cr=os.path.join(processed_dir, "{sub_id}", "{sub_id}_mask_cropped.nii.gz")
    params:
        img_dir=processed_dir,
        pad=pad
    log:
        "logs/{sub_id}.crop.log"
    shell:
        r"""
        set -euo pipefail
        export PYTHONPATH="${{PYTHONPATH:-}}:{PROJECT_ROOT}"
        mkdir -p logs {params.img_dir}/{wildcards.sub_id}
        conda run -n env-prep python scripts/cropping.py \
            --img_dir {params.img_dir} \
            --brain {input.head_fs} \
            --mask {input.mask_head} \
            --sub_id {wildcards.sub_id} \
            --pad {params.pad} \
            >> {log} 2>&1
        """


rule resample:
    input:
        brain_cr=os.path.join(processed_dir, "{sub_id}", "{sub_id}_brain_cropped.nii.gz"),
        mask_cr=os.path.join(processed_dir, "{sub_id}", "{sub_id}_mask_cropped.nii.gz")
    output:
        brain_re=os.path.join(processed_dir, "{sub_id}", "{sub_id}_brain_resampled.nii.gz"),
        mask_re=os.path.join(processed_dir, "{sub_id}", "{sub_id}_mask_resampled.nii.gz")
    params:
        img_dir=processed_dir,
        out_shape=out_shape
    log:
        "logs/{sub_id}.resample.log"
    shell:
        r"""
        set -euo pipefail
        export PYTHONPATH="${{PYTHONPATH:-}}:{PROJECT_ROOT}"
        mkdir -p logs {params.img_dir}/{wildcards.sub_id}
        conda run -n env-prep python scripts/resampling.py \
            --img_dir {params.img_dir} \
            --brain {input.brain_cr} \
            --mask {input.mask_cr} \
            --sub_id {wildcards.sub_id} \
            --out_shape {params.out_shape[0]} {params.out_shape[1]} {params.out_shape[2]} >> {log} 2>&1
        """

rule clip:
    input:
        brain_re=os.path.join(processed_dir, "{sub_id}", "{sub_id}_brain_resampled.nii.gz"),
        mask_re=os.path.join(processed_dir, "{sub_id}", "{sub_id}_mask_resampled.nii.gz")
    output:
        brain_cl=os.path.join(processed_dir, "{sub_id}", "{sub_id}_brain_clipped.nii.gz"),
        mask_cl=os.path.join(processed_dir, "{sub_id}", "{sub_id}_mask_clipped.nii.gz")
    params:
        img_dir=processed_dir,
        norm_perc=norm_perc
    log:
        "logs/{sub_id}.clip.log"
    shell:
        r"""
        set -euo pipefail
        export PYTHONPATH="${{PYTHONPATH:-}}:{PROJECT_ROOT}"
        mkdir -p logs {params.img_dir}/{wildcards.sub_id}
        conda run -n env-prep python scripts/clipping.py \
            --img_dir {params.img_dir} \
            --brain {input.brain_re} \
            --mask {input.mask_re} \
            --sub_id {wildcards.sub_id} \
            --norm_perc {params.norm_perc[0]} {params.norm_perc[1]} >> {log} 2>&1
        """

rule normalize:
    input:
        brain_cl=os.path.join(processed_dir, "{sub_id}", "{sub_id}_brain_clipped.nii.gz"),
        mask_cl=os.path.join(processed_dir, "{sub_id}", "{sub_id}_mask_clipped.nii.gz")
    output:
        brain_no=os.path.join(processed_dir, "{sub_id}", "{sub_id}_brain.nii.gz"),
        mask_no=os.path.join(processed_dir, "{sub_id}", "{sub_id}_mask.nii.gz")
    params:
        img_dir=processed_dir,
        norm_range=norm_range
    log:
        "logs/{sub_id}.normalize.log"
    shell:
        r"""
        set -euo pipefail
        export PYTHONPATH="${{PYTHONPATH:-}}:{PROJECT_ROOT}"
        mkdir -p logs {params.img_dir}/{wildcards.sub_id}
        conda run -n env-prep python scripts/normalization.py \
            --img_dir {params.img_dir} \
            --brain {input.brain_cl} \
            --mask {input.mask_cl} \
            --sub_id {wildcards.sub_id} \
            --norm_range {params.norm_range[0]} {params.norm_range[1]} \
            > {log} 2>&1
        """

rule dataset:
    input:
        brain_no=os.path.join(processed_dir, "{sub_id}", "{sub_id}_brain.nii.gz"),
        mask_no=os.path.join(processed_dir, "{sub_id}", "{sub_id}_mask.nii.gz")
    output:
        brain_dataset=os.path.join(dataset_dir, "image", "{sub_id}_brain.nii.gz"),
        mask_dataset=os.path.join(dataset_dir, "mask", "{sub_id}_mask.nii.gz")
    params:
        datasetdir=dataset_dir
    log:
        "logs/{sub_id}.dataset.log"
    shell:
        r"""
        set -euo pipefail
        mkdir -p {params.datasetdir}/image 
        mkdir -p {params.datasetdir}/mask
        cp {input.brain_no} {output.brain_dataset}
        cp {input.mask_no} {output.mask_dataset}
        """