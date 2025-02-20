# %% imports
import itertools as itt
import os
import pickle as pkl
import warnings

import holoviews as hv
import numpy as np
import xarray as xr
import yaml
from pydantic.v1.utils import deep_update

from routine.coregistration import apply_tx, estimate_tranform, process_temp, thres_roi
from routine.io import load_dataset
from routine.plotting import plot_ims, plotA_contour

hv.extension("bokeh")

IN_DPATH = "./data/full/"
IN_SS_CSV = "./data/full/sessions.csv"
IN_PARAM_PATH = "./params/"
PARAM_SKIP_EXISTING = True
PARAM_FLIP_ROI = True
PARAM_ROI_THRES = 0.8
OUT_PATH = "./intermediate/co-registration"
FIG_PATH = "./figs/co-registration"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


# %% load and coregistration
temp_out = os.path.join(OUT_PATH, "templates")
tx_out = os.path.join(OUT_PATH, "transform")
figpath = os.path.join(FIG_PATH, "templates")
os.makedirs(temp_out, exist_ok=True)
os.makedirs(tx_out, exist_ok=True)
os.makedirs(figpath, exist_ok=True)
for (anm, ss), ds, ssrow in load_dataset(
    IN_SS_CSV, IN_DPATH, load_rois=False, load_specs=False
):
    im_ms, im_conf = ds["im_ms"], ds["im_conf"]
    dsname = "{}-{}".format(anm, ss)
    if PARAM_SKIP_EXISTING and os.path.exists(
        os.path.join(temp_out, "{}.nc".format(dsname))
    ):
        print("skipping {}".format(dsname))
        continue
    param_files = ssrow["param"].split(";")
    param = dict()
    for pfile in param_files:
        with open(os.path.join(IN_PARAM_PATH, pfile)) as pf:
            param = deep_update(param, yaml.safe_load(pf))
    im_ms_ps = xr.apply_ufunc(
        process_temp,
        im_ms,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        kwargs=param["process_ms"],
    ).rename("ms-ps")
    im_conf_ps = xr.apply_ufunc(
        process_temp,
        im_conf,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        kwargs=param["process_conf"],
    ).rename("conf-ps")
    tx, tx_exh, param_df = estimate_tranform(
        im_ms_ps.data,
        im_conf_ps.data,
        scal_init=param["scal_init"],
        scal_stp=param["scal_stp"],
        scal_nstp=param["scal_nstp"],
        trans_stp=param["trans_stp"],
        trans_nstp=param["trans_nstp"],
        ang_stp=np.deg2rad(param["ang_stp"]),
        ang_nstp=param["ang_nstp"],
        lr=1,
    )
    im_ms_exh = xr.DataArray(
        apply_tx(im_ms, tx_exh, ref=im_conf.data, fill=np.nan),
        dims=["height", "width"],
        coords={"height": im_conf.coords["height"], "width": im_conf.coords["width"]},
        name="ms-exh",
    )
    ps_ms_exh = xr.DataArray(
        apply_tx(im_ms_ps, tx_exh, ref=im_conf_ps, fill=np.nan),
        dims=["height", "width"],
        coords={"height": im_conf.coords["height"], "width": im_conf.coords["width"]},
        name="ps-exh",
    )
    im_ms_reg = xr.DataArray(
        apply_tx(im_ms, tx, ref=im_conf.data, fill=np.nan),
        dims=["height", "width"],
        coords={"height": im_conf.coords["height"], "width": im_conf.coords["width"]},
        name="ms-reg",
    )
    ps_ms_reg = xr.DataArray(
        apply_tx(im_ms_ps, tx, ref=im_conf_ps, fill=np.nan),
        dims=["height", "width"],
        coords={"height": im_conf.coords["height"], "width": im_conf.coords["width"]},
        name="ps-reg",
    )
    ps_diff = (im_conf_ps - ps_ms_reg.fillna(0)).rename("ps-diff")
    im_diff = (im_conf - im_ms_reg.fillna(0)).rename("diff")
    fig = plot_ims(
        [im_ms, im_ms_ps, im_conf, im_conf_ps, im_ms_exh, im_ms_reg, ps_diff, im_diff],
        facet_col_wrap=4,
        norm=True,
    )
    fig.write_html(os.path.join(figpath, "{}.html".format(dsname)))
    ds = xr.merge(
        [
            im_ms,
            im_conf,
            im_ms_ps,
            im_conf_ps,
            im_ms_reg,
            ps_ms_reg,
            im_ms_exh,
            ps_ms_exh,
        ]
    )
    ds.to_netcdf(os.path.join(temp_out, "{}.nc".format(dsname)))
    with open(os.path.join(tx_out, "tx-{}.pkl".format(dsname)), "wb") as tx_file:
        pkl.dump(tx, tx_file)
    print(
        "data: {}, scale: {}, angle: {}, shift: {}".format(
            dsname, 1 / tx.GetScale(), np.rad2deg(tx.GetAngle()), tx.GetTranslation()
        )
    )

# %% transform roi
outpath = os.path.join(OUT_PATH, "rois")
figpath = os.path.join(FIG_PATH, "rois")
os.makedirs(outpath, exist_ok=True)
os.makedirs(figpath, exist_ok=True)
for (anm, ss), ds, ssrow in load_dataset(IN_SS_CSV, IN_DPATH, flip_rois=PARAM_FLIP_ROI):
    # load data
    dsname = "{}-{}".format(anm, ss)
    if PARAM_SKIP_EXISTING and os.path.exists(
        os.path.join(OUT_PATH, "rois", "{}.nc".format(dsname))
    ):
        continue
    reg_ds = xr.open_dataset(
        os.path.join(OUT_PATH, "templates", "{}.nc".format(dsname))
    )
    im_ms = reg_ds["ms-raw"].dropna("height", how="all").dropna("width", how="all")
    im_conf = reg_ds["conf-raw"].dropna("height", how="all").dropna("width", how="all")
    rois = ds["rois"]
    rois = xr.apply_ufunc(
        thres_roi,
        rois,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        vectorize=True,
        kwargs={"th": PARAM_ROI_THRES},
    )
    # transform roi
    with open(
        os.path.join(OUT_PATH, "transform", "tx-{}.pkl".format(dsname)), "rb"
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
    rois = rois.rename({"height_new": "height", "width_new": "width"}).assign_coords(
        {"height": im_conf.coords["height"], "width": im_conf.coords["width"]}
    )
    nempty = (rois.max(["height", "width"]) == 0).sum().item()
    if nempty > 0:
        warnings.warn("{} ROIs empty in dataset {}".format(nempty, dsname))
    # plot rois overlay
    fig = plotA_contour(
        im=im_conf,
        A=rois.rename(roi_id="unit"),
        im_opts={
            "frame_width": 400,
            "aspect": im_conf.sizes["width"] / im_conf.sizes["height"],
            "cmap": "gray",
        },
    )
    hv.save(fig, os.path.join(figpath, "{}.html".format(dsname)))
    rois.to_dataset().to_netcdf(os.path.join(outpath, "{}.nc".format(dsname)))

# %% compute correlation across ds
for ds1_name, ds2_name in itt.product(DS.keys(), repeat=2):
    ds1 = xr.open_dataset(os.path.join(OUT_PATH, "{}.nc".format(ds1_name)))
    ds2 = xr.open_dataset(os.path.join(OUT_PATH, "{}.nc".format(ds2_name)))
    im1 = ds1["ms-reg"].dropna("width", how="all").dropna("height", how="all")
    im2 = ds2["conf-raw"].dropna("width", how="all").dropna("height", how="all")
    if (
        im1.sizes["height"] == im2.sizes["height"]
        and im1.sizes["width"] == im2.sizes["width"]
    ):
        corr = np.corrcoef(np.array(im1).reshape(-1), np.array(im2).reshape(-1))[0, 1]
        print("{}-{}: {:.3f}".format(ds1_name, ds2_name, corr))
