# **ADNI_preprocessing**

This repository contains scripts and utilities for preprocessing ADNI MRI data.

Purpose of this work is to provide a reproducible pipeline for preprocessing of T1 whole head volumes :man:.

## Processing steps
- **FreeSurfer** *autorecon1* to standardize the images
- **Head extraction** 
- **Cropping** to its mask countures with padding and kkeping cubic shape according to min crop
- **Resampling** to fixed shape [*n*, *m*, *l*]
- **Intensities clipping** to certain percentages [*lower*, *higher*] and **normalization** to a fixed range [*min*, *max*] 


## Repo structure
- `scripts/` preprocessing scripts
    - `head_extraction_mri` extraction scripts (not mine)
    - `cropping` cropping to cubic box
    - `resmpling` spatial resampling 
    - `normalization` intensity clipping and normalization
    - `qc` plots of volume's slices and intensity histograms of images before and after preprocessing

- `utils/` helper modules 
    - `utils_preprocessing` preprocessing itself: resampling, clipping, normalization, ...
    - `utils_QC`quality control (QC) plots

- `config_pre.yaml` configuration file with directories, subjects' range and processing parameters


## Data :file_folder:
Data not included. ADNI data requires access approval. Provide path to downloaded ADNI files (subjects' images and *.csv file ) when running scripts.

###
## Note
- This preprocessing routine was used to provide a coherent dataset for the generation of synthetic brain and then whole head volumes. :brain:

## Usage exaples 

### Configuration

Modify `config_pre.yaml` with your info.

### snakemake :snake:

**dry run** with print command 

```bash
snakemake -n -p
```

**run all rules** :running_woman:
```bash
snakemake --use-conda --cores 50 --rerun-incomplete --latency-wait 900 --keep-going -p 
```

**single script** e.g. resampling

```bash
python scripts/resampling.py --output_dir --sub_dir XXX_S_XXX
```

## Requirements

TODO :see_no_evil: :hear_no_evil: :speak_no_evil: