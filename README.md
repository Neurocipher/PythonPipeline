# Neurocipher Python Pipeline

This is the python version of the neurocipher pipeline as described in [Functional imaging of nine distinct neuronal populations under a miniscope in freely behaving animals](https://www.biorxiv.org/content/10.1101/2023.12.13.571122v1).
It's currently under active development.

## Quick Start

To reproduce the results in the paper, or to use neurocipher pipeline on your own data, follow these quick start steps:

1. Setup conda environment: `conda env create -n neurocipher -f environments/generic.yml`
2. Obtain and organize your data under the `data` folder and make sure the [`DS`](https://github.com/Neurocipher/PythonPipeline/blob/b9491efe702e68f7ab4c6ab6154411fd64348239/00.co-registration.py#L13) variables reflects path to your data.
   Your dataset should contain:
   1. A `.tif` file of confocal image used for registration ([`DS[dsname]["conf"]`](https://github.com/Neurocipher/PythonPipeline/blob/b9491efe702e68f7ab4c6ab6154411fd64348239/00.co-registration.py#L15)),
   2. A `.tif` file of miniscope image used for registration ([`DS[dsname]["ms"]`](https://github.com/Neurocipher/PythonPipeline/blob/b9491efe702e68f7ab4c6ab6154411fd64348239/00.co-registration.py#L16)),
   3. `.czi` files from confocal for spectrum extraction. Their file names should contains strings indicating the imaging wavelength as specified by the [`specs`](https://github.com/Neurocipher/PythonPipeline/blob/b9491efe702e68f7ab4c6ab6154411fd64348239/routine/io.py#L57) argument,
   4. `ROIs.mat` containing the miniscope ROIs for spectrum extraction.
3. Run `00.co-registration.py`, `01.spectrum_extraction.py`, `02.fluoro_identification.py` in sequential order. The output cell mapping can be found in `output` folder. Various plots can be found under `figs` folder.
