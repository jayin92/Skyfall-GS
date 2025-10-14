# Evaluation Results

This directory contains the evaluation data, including ground truth videos and rendered videos from other methods.

## Downloading the Results

The evaluation data can be downloaded from the following link:

[Download Evaluation Data](https://drive.google.com/drive/folders/1hSFe9yGOwJCLBK7ZLHB-49_x73Ebk_VV?usp=drive_link)

After downloading, you can place the contents in this directory.

## Directory Structure

The directory is structured as follows:

```
results_eval/
├── data_eval_JAX/
│   ├── JAX_004/
│   │   ├── corgs/
│   │   ├── eogs/
│   │   ├── GT/
│   │   ├── mip-splatting/
│   │   ├── ours_stage1/
│   │   ├── ours_stage2/
│   │   └── sat-nerf/
│   ├── JAX_068/
│   │   └── ...
│   ├── JAX_214/
│   │   └── ...
│   └── JAX_260/
│       └── ...
└── data_eval_NYC/
    ├── NYC_004/
    │   ├── citydreamer/
    │   ├── corgs/
    │   ├── gaussiancity/
    │   ├── GT/
    │   ├── ours_stage1/
    │   └── ours_stage2/
    ├── NYC_010/
    │   └── ...
    ├── NYC_219/
    │   └── ...
    └── NYC_336/
        └── ...
```