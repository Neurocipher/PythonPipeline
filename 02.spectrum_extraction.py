# %% imports
import os

import cv2
import holoviews as hv
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from routine.io import load_spectif
from routine.plotting import plotA_contour
from routine.utilities import normalize

IN_DPATH = "./data/full/"
IN_SS_CSV = "./data/full/sessions.csv"
IN_ROI_PATH = "./intermediate/cross-registration/rois/"
OUT_PATH = "./intermediate/spectrum"
FIG_PATH = "./figs/spectrum"
PARAM_MED_WND = 3
PARAM_SUMZ = False
PARAM_SKIP_EXISTING = False

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)
hv.extension("bokeh")


# %% load data and extract spectrums
ss_csv = pd.read_csv(IN_SS_CSV).set_index("animal")
roi_files = list(filter(lambda f: f.endswith(".nc"), os.listdir(IN_ROI_PATH)))
for r in tqdm(roi_files):
    # load data
    rois = xr.open_dataset(os.path.join(IN_ROI_PATH, r))["rois"]
    try:
        anm, ss = rois.coords["animal"].item(), rois.coords["session"].item()
        dsname = "{}-{}".format(anm, ss)
    except KeyError:
        anm = rois.coords["animal"].item()
        dsname = anm
    spec = ss_csv.loc[anm]["specs"].unique().item()
    ims_conf = load_spectif(
        os.path.dirname(os.path.join(IN_DPATH, spec)), os.path.basename(spec)
    )
    # process czi
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
