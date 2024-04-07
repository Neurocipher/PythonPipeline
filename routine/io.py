import numpy as np
import tifffile as tiff
import xarray as xr

from .utilities import normalize


def read_templates(im_ms, im_conf, flip=True, norm=True):
    im_ms = np.array(tiff.imread(im_ms)).astype(float)
    im_conf = np.array(tiff.imread(im_conf)).astype(float)
    if flip:
        im_ms = np.flip(im_ms)
    if norm:
        im_ms = normalize(im_ms)
        im_conf = normalize(im_conf)
    return (
        xr.DataArray(
            im_ms,
            dims=["height", "width"],
            coords={
                "height": np.arange(im_ms.shape[0]),
                "width": np.arange(im_ms.shape[1]),
            },
            name="ms-raw",
        ),
        xr.DataArray(
            im_conf,
            dims=["height", "width"],
            coords={
                "height": np.arange(im_conf.shape[0]),
                "width": np.arange(im_conf.shape[1]),
            },
            name="conf-raw",
        ),
    )
