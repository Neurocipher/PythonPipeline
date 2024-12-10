# %% imports
import os
import pickle as pkl
import warnings

import cv2
import holoviews as hv
import xarray as xr

from routine.coregistration import apply_tx
from routine.io import load_dataset
from routine.plotting import plotA_contour
from routine.utilities import normalize

IN_DPATH = "./data/full/"
IN_SS_CSV = "./data/full/sessions.csv"
IN_REG_PATH = "./intermediate/co-registration/"
OUT_PATH = "./intermediate/spectrum"
FIG_PATH = "./figs/spectrum"
PARAM_MED_WND = 3
PARAM_SUMZ = False
PARAM_TRANSFORM_ROI = True
PARAM_FLIP_ROI = True
PARAM_SKIP_EXISTING = False

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)
hv.extension("bokeh")


# %% load data and extract spectrums
for (anm, ss), ds, ssrow in load_dataset(IN_DPATH, IN_SS_CSV, flip_rois=PARAM_FLIP_ROI):
    # load data
    dsname = "{}-{}".format(anm, ss)
    if PARAM_SKIP_EXISTING and os.path.exists(
        os.path.join(OUT_PATH, "{}.nc".format(dsname))
    ):
        continue
    reg_ds = xr.open_dataset(os.path.join(IN_REG_PATH, "{}.nc".format(dsname)))
    im_ms = reg_ds["ms-raw"].dropna("height", how="all").dropna("width", how="all")
    im_conf = reg_ds["conf-raw"].dropna("height", how="all").dropna("width", how="all")
    rois = ds["rois"]
    ims_conf = ds["specs"]
    # transform roi
    if PARAM_TRANSFORM_ROI:
        with open(
            os.path.join(IN_REG_PATH, "tx-{}.pkl".format(dsname)), "rb"
        ) as tx_file:
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
    ims_conf = xr.apply_ufunc(
        cv2.medianBlur,
        ims_conf,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"ksize": PARAM_MED_WND},
    )
    if PARAM_SUMZ:
        ims_conf = ims_conf.sum("z").compute()
    # plot rois overlay
    ims_chns = ims_conf.groupby("channel_group").max("channel")
    im_dict = dict()
    for chn, chn_dat in ims_chns.groupby("channel_group", squeeze=False):
        im = plotA_contour(
            im=chn_dat.rename(chn).squeeze(),
            A=rois.rename(roi_id="unit"),
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
    spec_ds = xr.merge(
        [
            specs,
            specs_norm,
            ims_conf.rename("ims_conf"),
            ims_chns.rename("ims_chns"),
            rois.rename("rois"),
        ]
    )
    spec_ds.to_netcdf(os.path.join(OUT_PATH, "{}.nc".format(dsname)))
