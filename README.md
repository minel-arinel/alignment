# alignment
Spatial registration of calcium imaging datasets to reference images.

## Setup

To create a conda environment using the provided `environment.yml` file, follow these steps:

1. Open a terminal.
2. Navigate to the directory containing the `environment.yml` file:
    ```sh
    cd alignment
    ```
3. Create the conda environment:
    ```sh
    conda env create -f environment.yml
    ```
4. Activate the newly created environment:
    ```sh
    conda activate alignment
    ```

You are now ready to use the `alignment` environment.

### Elastix (optional)

Piecewise registration in `brain_alignment_from_tiff_masks.ipynb` uses **image blackout** (zero outside each band/ROI) and SimpleITK's Elastix wrapper by default—no standalone `elastix` binary required.

To use metric masks via the CLI (`-fMask` / `-mMask`), call `register_image2(..., use_elastix_cli=True)` and install binaries into the repo (once per machine):

```sh
bash scripts/install_elastix.sh
```

This unpacks to `tools/elastix/` (gitignored). `registration.sitkalignment` auto-detects that path. You can also set `ELASTIX_EXECUTABLE` to an existing `elastix` binary.

## Note

The files in the `registration` folder are from the [Naumann Lab scopeslip repository](https://github.com/Naumann-Lab/scopeslip).
