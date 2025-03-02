# %% imports
import itertools as itt
import os
import pickle as pkl
import warnings

import colorcet as cc
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import xarray as xr
import yaml
from pydantic.v1.utils import deep_update

from routine.coregistration import apply_tx, estimate_tranform, process_temp, thres_roi
from routine.io import load_dataset
from routine.plotting import im_overlay, plot_ims, plotA_contour
from routine.utilities import compute_corr, normalize

hv.extension("bokeh")

IN_DPATH = "./data/full/"
IN_SS_CSV = "./data/full/sessions.csv"
IN_PARAM_PATH = "./params/"
PARAM_SKIP_EXISTING = False
PARAM_FLIP_ROI = True
PARAM_ROI_THRES = 0.95
OUT_PATH = "./intermediate/co-registration"
FIG_PATH = "./figs/co-registration"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


# %% load and coregistration
temp_out = os.path.join(OUT_PATH, "templates")
tx_out = os.path.join(OUT_PATH, "transform")
os.makedirs(temp_out, exist_ok=True)
os.makedirs(tx_out, exist_ok=True)
for (anm, ss), ds, ssrow in load_dataset(
    IN_SS_CSV, IN_DPATH, load_rois=False, load_specs=False
):
    ms_raw, conf_raw = ds["im_ms"], ds["im_conf"]
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
    if param["process_ms"] is not None:
        ms_ps = xr.apply_ufunc(
            process_temp,
            ms_raw,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            kwargs=param["process_ms"],
        ).rename("ms-ps")
    else:
        ms_ps = ms_raw.rename("ms-ps")
    if param["process_conf"] is not None:
        conf_ps = xr.apply_ufunc(
            process_temp,
            conf_raw,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            kwargs=param["process_conf"],
        ).rename("conf-ps")
    else:
        conf_ps = conf_raw.rename("conf-ps")
    tx, tx_exh, param_df = estimate_tranform(
        ms_ps.data,
        conf_ps.data,
        scal_init=param["scal_init"],
        scal_stp=param["scal_stp"],
        scal_nstp=param["scal_nstp"],
        trans_stp=param["trans_stp"],
        trans_nstp=param["trans_nstp"],
        ang_stp=np.deg2rad(param["ang_stp"]),
        ang_nstp=param["ang_nstp"],
        lr=1,
    )
    ms_exh = xr.DataArray(
        apply_tx(ms_raw, tx_exh, ref=conf_raw.data, fill=np.nan),
        dims=["height", "width"],
        coords={"height": conf_raw.coords["height"], "width": conf_raw.coords["width"]},
        name="ms-exh",
    )
    ps_ms_exh = xr.DataArray(
        apply_tx(ms_ps, tx_exh, ref=conf_ps, fill=np.nan),
        dims=["height", "width"],
        coords={"height": conf_raw.coords["height"], "width": conf_raw.coords["width"]},
        name="ps-exh",
    )
    ms_reg = xr.DataArray(
        apply_tx(ms_raw, tx, ref=conf_raw.data, fill=np.nan),
        dims=["height", "width"],
        coords={"height": conf_raw.coords["height"], "width": conf_raw.coords["width"]},
        name="ms-reg",
    )
    ps_reg = xr.DataArray(
        apply_tx(ms_ps, tx, ref=conf_ps, fill=np.nan),
        dims=["height", "width"],
        coords={"height": conf_raw.coords["height"], "width": conf_raw.coords["width"]},
        name="ps-reg",
    )
    ds = xr.merge(
        [
            ms_raw,
            conf_raw,
            ms_ps,
            conf_ps,
            ms_reg,
            ps_reg,
            ms_exh,
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

# %% plot coregistration results
temp_out = os.path.join(OUT_PATH, "templates")
figpath = os.path.join(FIG_PATH, "templates")
os.makedirs(figpath, exist_ok=True)
for (anm, ss), ds, ssrow in load_dataset(
    IN_SS_CSV, IN_DPATH, load_rois=False, load_specs=False
):
    # load data
    dsname = "{}-{}".format(anm, ss)
    reg_ds = xr.open_dataset(os.path.join(temp_out, "{}.nc".format(dsname)))
    ms_raw, ms_ps, conf_raw, conf_ps, ms_exh, ms_reg, ps_reg = (
        reg_ds["ms-raw"],
        reg_ds["ms-ps"],
        reg_ds["conf-raw"],
        reg_ds["conf-ps"],
        reg_ds["ms-exh"],
        reg_ds["ms-reg"],
        reg_ds["ps-reg"],
    )
    ps_diff = (conf_ps - ps_reg.fillna(0)).rename("ps-diff")
    im_diff = (conf_raw - ms_reg.fillna(0)).rename("diff")
    # html figure
    fig = plot_ims(
        [ms_raw, ms_ps, conf_raw, conf_ps, ms_exh, ms_reg, ps_diff, im_diff],
        facet_col_wrap=4,
        norm=True,
    )
    fig.write_html(os.path.join(figpath, "{}.html".format(dsname)))
    # static figure
    scale = min(
        conf_raw.sizes["height"] / ms_raw.sizes["height"],
        conf_raw.sizes["width"] / ms_raw.sizes["width"],
    )
    tx = sitk.ScaleTransform(2, 1 / scale * np.ones(2))
    ms_scale = xr.DataArray(
        apply_tx(ms_raw, tx, fill=np.nan, ref=conf_raw.data),
        dims=["height", "width"],
        coords={"height": conf_raw.coords["height"], "width": conf_raw.coords["width"]},
        name="ms-scaled",
    )
    fig, axs = plt.subplots(4, 2, figsize=(5, 10))
    cm_ms = cc.cm["kg"]
    cm_conf = cc.cm["kr"]
    cm_diff = cc.cm["cwr"]
    corr_scale, corr_exh, corr_reg = (
        compute_corr(ms_scale, conf_raw),
        compute_corr(ms_exh, conf_raw),
        compute_corr(ms_reg, conf_raw),
    )
    ms_scale, ms_exh, ms_reg, conf_raw = (
        normalize(-ms_scale, (0.01, 0.99)),
        normalize(-ms_exh, (0.01, 0.99)),
        normalize(-ms_reg, (0.01, 0.99)),
        normalize(-conf_raw, (0.05, 0.95)),
    )
    axs[0, 0].set_title("Miniscope Image")
    axs[0, 0].imshow(ms_scale, cmap=cm_ms)
    axs[0, 1].set_title("Confocal Image")
    axs[0, 1].imshow(conf_raw, cmap=cm_conf)
    axs[1, 0].set_title("Raw Overlay\ncorr: {:.2f}".format(corr_scale))
    axs[1, 0].imshow(im_overlay(ms_scale, conf_raw, cm_ms, cm_conf))
    axs[1, 0].contour(np.isnan(ms_scale), colors="gray")
    axs[1, 1].set_title("Raw Diff")
    axs[1, 1].imshow(ms_scale - conf_raw, cmap=cm_diff)
    axs[2, 0].set_title("First Pass Overlay\ncorr: {:.2f}".format(corr_exh))
    axs[2, 0].imshow(im_overlay(ms_exh, conf_raw, cm_ms, cm_conf))
    axs[2, 0].contour(np.isnan(ms_exh), colors="gray")
    axs[2, 1].set_title("First Pass Diff")
    axs[2, 1].imshow(ms_exh - conf_raw, cmap=cm_diff)
    axs[3, 0].set_title("Second Pass Overlay\ncorr: {:.2f}".format(corr_reg))
    axs[3, 0].imshow(im_overlay(ms_reg, conf_raw, cm_ms, cm_conf))
    axs[3, 0].contour(np.isnan(ms_reg), colors="gray")
    axs[3, 1].set_title("Second Pass Diff")
    axs[3, 1].imshow(ms_reg - conf_raw, cmap=cm_diff)
    for ax in axs.flatten():
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(
        os.path.join(figpath, "{}.svg".format(dsname)), dpi=500, bbox_inches="tight"
    )
    plt.close(fig)

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
    ms_raw = reg_ds["ms-raw"].dropna("height", how="all").dropna("width", how="all")
    conf_raw = reg_ds["conf-raw"].dropna("height", how="all").dropna("width", how="all")
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
        kwargs={"tx": tx, "ref": conf_raw},
    )
    rois = rois.rename({"height_new": "height", "width_new": "width"}).assign_coords(
        {"height": conf_raw.coords["height"], "width": conf_raw.coords["width"]}
    )
    nempty = (rois.max(["height", "width"]) == 0).sum().item()
    if nempty > 0:
        warnings.warn("{} ROIs empty in dataset {}".format(nempty, dsname))
    # plot rois overlay
    fig = plotA_contour(
        im=conf_raw,
        A=rois.rename(roi_id="unit"),
        im_opts={
            "frame_width": 400,
            "aspect": conf_raw.sizes["width"] / conf_raw.sizes["height"],
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
