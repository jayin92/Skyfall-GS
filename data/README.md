# Datasets

This directory is the default location for the datasets required to train the Skyfall-GS.

## Downloading the Datasets

The JAX and NYC datasets are available for download from Google Drive.

1.  **Download the zip files:**

    [Download from Google Drive](https://drive.google.com/drive/folders/1Uugwpf7n5fj7k4UJRBuKUyrmkYcDRScQ?usp=drive_link)

2.  **Unzip the datasets into this `data/` directory:**

    ```bash
    unzip datasets_JAX.zip
    unzip datasets_NYC.zip
    ```

## Directory Structure

After unzipping, the directory structure should look like this:

```
data/
├── datasets_JAX/
│   ├── JAX_004
│   ├── JAX_068
│   ├── JAX_164
│   ├── JAX_168
│   ├── JAX_175
│   ├── JAX_214
│   ├── JAX_260
│   └── JAX_264
└── datasets_NYC/
    ├── NYC_004
    ├── NYC_010
    ├── NYC_219
    └── NYC_336
```