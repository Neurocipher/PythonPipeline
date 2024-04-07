import numpy as np
import plotly.express as px
import xarray as xr

from .utilities import normalize


def plot_ims(ims, norm=True, **kwargs):
    if norm:
        ims = [
            (
                xr.apply_ufunc(
                    normalize,
                    im,
                    input_core_dims=[["height", "width"]],
                    output_core_dims=[["height", "width"]],
                )
                * 255
            ).astype(np.uint8)
            for im in ims
        ]
    ims = [im.assign_coords(im_name=im.name) for im in ims]
    return px.imshow(
        xr.concat(ims, "im_name"), facet_col="im_name", aspect="equal", **kwargs
    )
