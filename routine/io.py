import os
import re
import warnings

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
import xarray as xr
from aicsimageio import AICSImage
from scipy.io import loadmat
from tqdm.auto import tqdm

from .utilities import normalize, split_path


def load_dataset(
    ss_csv,
    dpath: str = None,
    id_cols=["animal", "session"],
    load_temps=True,
    load_rois=True,
    load_specs=True,
    flip_rois=False,
    rois_accept_only=False,
):
    ssdf = pd.read_csv(ss_csv).set_index(id_cols)
    for idxs, ssrow in tqdm(ssdf.iterrows(), total=len(ssdf)):
        if dpath is None:
            yield idxs, ssrow
        else:
            ret_ds = dict()
            if load_temps:
                if pd.isnull(ssrow["temp_ms"]) or pd.isnull(ssrow["temp_conf"]):
                    warnings.warn(
                        "Cannot load templates for {}. Skipping".format(str(idxs))
                    )
                    continue
                im_ms, im_conf = load_templates(
                    os.path.join(dpath, ssrow["temp_ms"]),
                    os.path.join(dpath, ssrow["temp_conf"]),
                )
                ret_ds["im_ms"] = im_ms
                ret_ds["im_conf"] = im_conf
            if load_rois:
                if pd.isnull(ssrow["rois"]):
                    warnings.warn("Cannot load ROIs for {}. Skipping".format(str(idxs)))
                    continue
                dname, basename = split_path(os.path.join(dpath, ssrow["rois"]))
                rois = load_roitif(dname, basename, flip=flip_rois)
                if rois_accept_only:
                    roidf = pd.read_csv(os.path.join(dname, "CellTraces-props.csv"))
                    rois = rois.sel(
                        roi_id=np.array(roidf.set_index("Name")["Status"] == "accepted")
                    )
                ret_ds["rois"] = rois
            if load_specs:
                if pd.isnull(ssrow["specs"]):
                    warnings.warn(
                        "Cannot load spectrum for {}. Skipping".format(str(idxs))
                    )
                    continue
                dname, basename = split_path(os.path.join(dpath, ssrow["specs"]))
                ret_ds["specs"] = load_spectif(dname, basename)
            yield idxs, ret_ds, ssrow


def load_templates(im_ms, im_conf, flip=False, norm=True):
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


def load_roitif(dpath, pat, flip=False):
    rois = []
    for tf in os.listdir(dpath):
        ma = re.search(pat, tf)
        if ma is not None:
            roi = np.array(tiff.imread(os.path.join(dpath, tf)))
            if flip:
                roi = np.flip(roi, axis=[0, 1])
            roi = xr.DataArray(
                roi,
                dims=["height", "width"],
                coords={
                    "height": np.arange(roi.shape[0]),
                    "width": np.arange(roi.shape[1]),
                    "roi_id": ma.group(1),
                },
            )
            rois.append(roi)
    try:
        return xr.concat(rois, "roi_id").sortby("roi_id").rename("rois")
    except ValueError:
        raise FileNotFoundError(
            "No valid ROIs found under {} with pattern {}".format(dpath, pat)
        )


def load_spectif(dpath, pat):
    specs = []
    for tf in os.listdir(dpath):
        ma = re.search(pat, tf)
        if ma is not None:
            chn = ma.group(1)
            spec = np.array(tiff.imread(os.path.join(dpath, tf)))
            spec = xr.DataArray(
                spec,
                dims=["channel", "height", "width"],
                coords={
                    "height": np.arange(spec.shape[1]),
                    "width": np.arange(spec.shape[2]),
                    "channel": [
                        "{}-{:0>2}".format(chn, c) for c in range(spec.shape[0])
                    ],
                    "channel_group": ("channel", [chn] * spec.shape[0]),
                },
            )
            specs.append(spec)
    try:
        return xr.concat(specs, "channel").sortby("channel").rename("specs")
    except ValueError:
        raise FileNotFoundError(
            "No valid spectrum tifs found under {} with pattern {}".format(dpath, pat)
        )


def load_refmat(matfile: str):
    mat = loadmat(matfile)
    return xr.DataArray(mat["hek"], dims=["fluo", "channel"]), xr.DataArray(
        mat["PD"], dims=["dist", "fluo"], coords={"dist": ["raw", "norm"]}
    )


def load_cellsmat(matfile: str):
    mat = loadmat(matfile)
    return xr.DataArray(mat["cells_line"], dims=["unit", "channel"]), xr.DataArray(
        mat["cells_n_line"], dims=["unit", "channel"]
    )


def load_czi(path, n_exclude=2, specs=["405", "488", "514", "561", "594", "639"]):
    czi_files = list(filter(lambda fn: fn.endswith(".czi"), os.listdir(path)))
    arr_ls = []
    for spec in specs:
        cur_czi = list(
            filter(lambda fn: spec in fn[:6], czi_files)
        )  # TODO: make filename parsing more intelligent
        assert len(cur_czi) == 1, "CZI files missing or duplicated for {}: {}".format(
            spec, cur_czi
        )
        cur_czi = os.path.join(path, cur_czi[0])
        cur_im = AICSImage(cur_czi)
        cur_arr = xr.DataArray(
            cur_im.get_image_dask_data("XYZC", T=0),
            dims=["width", "height", "z", "channel"],
        )
        if n_exclude:
            cur_arr = cur_arr.isel(z=slice(n_exclude, -n_exclude))
        cur_arr = cur_arr.assign_coords(
            width=np.arange(cur_arr.sizes["width"]),
            height=np.arange(cur_arr.sizes["height"]),
            z=np.arange(cur_arr.sizes["z"]),
            channel=[
                "{}-{:0>2}".format(spec, i) for i in range(cur_arr.sizes["channel"])
            ],
            channel_group=("channel", [spec] * cur_arr.sizes["channel"]),
        )
        arr_ls.append(cur_arr)
    return xr.concat(arr_ls, dim="channel")


def load_roimat(matfile: str, ref_im: xr.DataArray):
    mat = loadmat(matfile)
    rois = mat["ROIs"].squeeze()
    A = np.zeros((len(rois), ref_im.sizes["width"], ref_im.sizes["height"]))
    for ir, roi in enumerate(rois):
        # edges = np.zeros_like(A[ir, :, :])
        # edges[roi[:, 0], roi[:, 1]] = 1
        filled = cv2.drawContours(
            A[ir, :, :],
            [roi[:, ::-1].astype(np.int32)],
            contourIdx=-1,
            color=1,
            thickness=cv2.FILLED,
        )
        A[ir, :, :] = filled
    return xr.DataArray(
        A,
        dims=["unit", "width", "height"],
        coords={
            "unit": np.arange(A.shape[0]),
            "width": ref_im.coords["width"],
            "height": ref_im.coords["height"],
        },
    )
