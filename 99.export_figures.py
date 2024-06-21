# %% import
import os

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import xarray as xr

from routine.coregistration import apply_tx, remove_background
from routine.plotting import im_overlay, plotA_contour_mpl
from routine.utilities import compute_corr, normalize

DS = {
    "21272R": {
        "reg_ds": "./intermediate/co-registration/21272R.nc",
        "spec_ds": "./intermediate/spectrum/21272R.nc",
        "use_raw": False,
        # "back_wnd": (101, 101),
    }
}
FIG_PATH = "./figs/export"

# %% export plots
for dsname, dsdat in DS.items():
    # load data
    reg_ds = xr.open_dataset(dsdat["reg_ds"])
    spec_ds = xr.open_dataset(dsdat["spec_ds"])
    # plot registration
    fig_path = os.path.join(FIG_PATH, "{}".format(dsname))
    os.makedirs(fig_path, exist_ok=True)
    if dsdat["use_raw"]:
        ms_raw, ms_exh, ms_reg, conf = (
            reg_ds["ms-raw"].dropna("width", how="all").dropna("height", how="all"),
            reg_ds["ms-exh"],
            reg_ds["ms-reg"],
            reg_ds["conf-raw"],
        )
    else:
        ms_raw, ms_exh, ms_reg, conf = (
            reg_ds["ms-ps"].dropna("width", how="all").dropna("height", how="all"),
            reg_ds["ps-exh"],
            reg_ds["ps-reg"],
            reg_ds["conf-ps"],
        )
    back_wnd = dsdat.get("back_wnd")
    if back_wnd:
        ms_raw = xr.apply_ufunc(
            remove_background,
            ms_raw,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            kwargs={"back_wnd": back_wnd},
        )
        conf = xr.apply_ufunc(
            remove_background,
            conf,
            input_core_dims=[["height", "width"]],
            output_core_dims=[["height", "width"]],
            kwargs={"back_wnd": back_wnd},
        )
    scale = min(
        conf.sizes["height"] / ms_raw.sizes["height"],
        conf.sizes["width"] / ms_raw.sizes["width"],
    )
    tx = sitk.ScaleTransform(2, 1 / scale * np.ones(2))
    ms_scale = xr.DataArray(
        apply_tx(ms_raw, tx, fill=np.nan, ref=conf.data),
        dims=["height", "width"],
        coords={"height": conf.coords["height"], "width": conf.coords["width"]},
        name="ms-scaled",
    )
    fig, axs = plt.subplots(4, 2, figsize=(5, 10))
    cm_ms = cc.cm["kg"]
    cm_conf = cc.cm["kr"]
    cm_diff = cc.cm["cwr"]
    ms_scale, ms_exh, ms_reg, conf = (
        normalize(ms_scale, (0.01, 0.99)),
        normalize(ms_exh, (0.01, 0.99)),
        normalize(ms_reg, (0.01, 0.99)),
        normalize(conf, (0.85, 0.95)),
    )
    corr_scale, corr_exh, corr_reg = (
        compute_corr(ms_scale, conf),
        compute_corr(ms_exh, conf),
        compute_corr(ms_reg, conf),
    )
    axs[0, 0].set_title("Miniscope Image")
    axs[0, 0].imshow(ms_scale, cmap=cm_ms)
    axs[0, 1].set_title("Confocal Image")
    axs[0, 1].imshow(conf, cmap=cm_conf)
    axs[1, 0].set_title("Raw Overlay\ncorr: {:.2f}".format(corr_scale))
    axs[1, 0].imshow(im_overlay(ms_scale, conf, cm_ms, cm_conf))
    axs[1, 0].contour(np.isnan(ms_scale), colors="gray")
    axs[1, 1].set_title("Raw Diff")
    axs[1, 1].imshow(ms_scale - conf, cmap=cm_diff)
    axs[2, 0].set_title("First Pass Overlay\ncorr: {:.2f}".format(corr_exh))
    axs[2, 0].imshow(im_overlay(ms_exh, conf, cm_ms, cm_conf))
    axs[2, 0].contour(np.isnan(ms_exh), colors="gray")
    axs[2, 1].set_title("First Pass Diff")
    axs[2, 1].imshow(ms_exh - conf, cmap=cm_diff)
    axs[3, 0].set_title("Second Pass Overlay\ncorr: {:.2f}".format(corr_reg))
    axs[3, 0].imshow(im_overlay(ms_reg, conf, cm_ms, cm_conf))
    axs[3, 0].contour(np.isnan(ms_reg), colors="gray")
    axs[3, 1].set_title("Second Pass Diff")
    axs[3, 1].imshow(ms_reg - conf, cmap=cm_diff)
    for ax in axs.flatten():
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_path, "registration.svg"), dpi=500, bbox_inches="tight"
    )
    # plot rois
    fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
    rois, ims_chns = spec_ds["rois"], spec_ds["ims_chns"]
    exp_roi = 27
    axs[0].set_title("561nm channel")
    plotA_contour_mpl(
        rois,
        normalize(ims_chns.sel(channel_group="561"), (0.01, 0.999)),
        im_cmap=cc.cm["kr"],
        cnt_kws={"linewidths": 0.2, "colors": "gray"},
        ax=axs[0],
    )
    plotA_contour_mpl(
        rois.sel(unit=[exp_roi]),
        normalize(ims_chns.sel(channel_group="561"), (0.01, 0.999)),
        im_cmap=cc.cm["kr"],
        cnt_kws={"linewidths": 0.5, "colors": "royalblue"},
        ax=axs[0],
    )
    axs[1].set_title("594nm channel")
    plotA_contour_mpl(
        rois,
        normalize(ims_chns.sel(channel_group="594"), (0.01, 0.998)),
        im_cmap=cc.cm["kr"],
        cnt_kws={"linewidths": 0.2, "colors": "gray"},
        ax=axs[1],
    )
    plotA_contour_mpl(
        rois.sel(unit=[exp_roi]),
        normalize(ims_chns.sel(channel_group="594"), (0.01, 0.998)),
        im_cmap=cc.cm["kr"],
        cnt_kws={"linewidths": 0.5, "colors": "royalblue"},
        ax=axs[1],
    )
    for ax in axs.flatten():
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_path, "example_roi.svg"), dpi=500, bbox_inches="tight")
