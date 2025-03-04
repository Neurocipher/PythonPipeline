import cv2
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import xarray as xr
from matplotlib import cm

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


def plotA_contour(A: xr.DataArray, im: xr.DataArray, cmap=None, im_opts=None):
    im = hv.Image(im, ["width", "height"])
    if im_opts is not None:
        im = im.opts(**im_opts)
    im = im * hv.Path([])
    for uid in A.coords["unit"].values:
        curA = (np.array(A.sel(unit=uid)) > 0).astype(np.uint8)
        try:
            cnt = cv2.findContours(curA, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][
                0
            ].squeeze()
        except IndexError:
            continue
        if cnt.ndim > 1:
            cnt_scale = np.zeros_like(cnt)
            cnt_scale[:, 0] = A.coords["width"][cnt[:, 0]]
            cnt_scale[:, 1] = A.coords["height"][cnt[:, 1]]
            pth = hv.Path(cnt_scale.squeeze())
            if cmap is not None:
                pth = pth.opts(color=cmap[uid])
            im = im * pth
    return im


def im_overlay(imA, imB, cmapA, cmapB, brt_offset=0):
    im_pcolA = np.clip(
        cm.ScalarMappable(cmap=cmapA).to_rgba(normalize(np.nan_to_num(imA)))
        + brt_offset,
        0,
        1,
    )
    im_pcolB = np.clip(
        cm.ScalarMappable(cmap=cmapB).to_rgba(normalize(np.nan_to_num(imB)))
        + brt_offset,
        0,
        1,
    )
    return np.clip(im_pcolA + im_pcolB, 0, 1)


def plotA_contour_mpl(
    A: xr.DataArray,
    im: xr.DataArray = None,
    cmap=None,
    cnt_kws=dict(),
    im_cmap=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    if im is not None:
        ax.imshow(im, cmap=im_cmap)
    for uid in A.coords["unit"].values:
        curA = (np.array(A.sel(unit=uid)) > 0).astype(np.uint8)
        try:
            cnt = cv2.findContours(curA, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][
                0
            ].squeeze()
        except IndexError:
            continue
        if cnt.ndim > 1:
            cnt_scale = np.zeros_like(cnt)
            cnt_scale[:, 0] = A.coords["width"][cnt[:, 0]]
            cnt_scale[:, 1] = A.coords["height"][cnt[:, 1]]
            if cmap is not None:
                ax.plot(cnt_scale[:, 0], cnt_scale[:, 1], color=cmap[uid], **cnt_kws)
            else:
                ax.plot(cnt_scale[:, 0], cnt_scale[:, 1], **cnt_kws)
    return ax
