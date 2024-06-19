# %% imports
import os
import pickle as pkl
import warnings

import cv2
import holoviews as hv
import xarray as xr

from routine.coregistration import apply_tx
from routine.io import load_czi, load_roimat
from routine.plotting import plotA_contour
from routine.utilities import normalize

DS = {
    "21271R": {
        "ds_path": "./data/demo/21271R",
        "reg_ds": "./intermediate/co-registration/21271R.nc",
        "tx": "./intermediate/co-registration/tx-21271R.pkl",
        "transform_roi": False,
    },
    "21272R": {
        "ds_path": "./data/demo/21272R",
        "reg_ds": "./intermediate/co-registration/21272R.nc",
        "tx": "./intermediate/co-registration/tx-21272R.pkl",
        "transform_roi": False,
    },
    "25607": {
        "ds_path": "./data/demo/25607",
        "reg_ds": "./intermediate/co-registration/25607.nc",
        "tx": "./intermediate/co-registration/tx-25607.pkl",
        "transform_roi": True,
    },
}
OUT_PATH = "./intermediate/spectrum"
FIG_PATH = "./figs/spectrum"
PARAM_MED_WND = 3
PARAM_TRANSFORM_ROI = False

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)
hv.notebook_extension("bokeh")


# %% load data and extract spectrums
for dsname, dsdat in DS.items():
    # load rois
    reg_ds = xr.open_dataset(dsdat["reg_ds"])
    im_ms = reg_ds["ms-raw"].dropna("height", how="all").dropna("width", how="all")
    im_conf = reg_ds["conf-raw"].dropna("height", how="all").dropna("width", how="all")
    im_ref = im_ms if dsdat["transform_roi"] else im_conf
    rois = load_roimat(os.path.join(dsdat["ds_path"], "ROIs.mat"), im_ref)
    # transform roi
    if dsdat["transform_roi"]:
        with open(dsdat["tx"], "rb") as tx_file:
            tx = pkl.load(tx_file)
        rois = xr.apply_ufunc(
            apply_tx,
            rois,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height_new", "width_new"]],
            vectorize=True,
            kwargs={"tx": tx, "ref": im_conf},
        )
        rois = rois.rename(
            {"height_new": "height", "width_new": "width"}
        ).assign_coords(
            {"height": im_conf.coords["height"], "width": im_conf.coords["width"]}
        )
    nempty = (rois.max(["height", "width"]) == 0).sum().item()
    if nempty > 0:
        warnings.warn("{} ROIs empty in dataset {}".format(nempty, dsname))
    # load czi
    ims_conf = load_czi(dsdat["ds_path"])
    ims_conf = xr.apply_ufunc(
        cv2.medianBlur,
        ims_conf,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"ksize": PARAM_MED_WND},
    )
    ims_conf = ims_conf.sum("z").compute()
    # plot rois overlay
    ims_chns = ims_conf.groupby("channel_group").max("channel")
    im_dict = dict()
    for chn, chn_dat in ims_chns.groupby("channel_group", squeeze=False):
        im = plotA_contour(
            im=chn_dat.rename(chn).squeeze(),
            A=rois,
            im_opts={
                "frame_width": 400,
                "aspect": chn_dat.sizes["width"] / chn_dat.sizes["height"],
                "cmap": "gray",
            },
        )
        im_dict[chn] = im
    fig = hv.NdLayout(im_dict, "channel")
    hv.save(fig, os.path.join(FIG_PATH, "{}.html".format(dsname)))
    # extract spectrum
    specs = (rois.dot(ims_conf) / rois.sum(["height", "width"])).rename("spec_raw")
    specs_norm = xr.apply_ufunc(
        normalize,
        specs,
        input_core_dims=[["channel"]],
        output_core_dims=[["channel"]],
        vectorize=True,
    ).rename("spec_norm")
    spec_ds = xr.merge([specs, specs_norm])
    spec_ds.to_netcdf(os.path.join(OUT_PATH, "{}.nc".format(dsname)))
