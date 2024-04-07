# %% imports
import os

import numpy as np
import xarray as xr

from routine.coregistration import apply_tx, est_sim, process_temp
from routine.io import read_templates
from routine.plotting import plot_ims

DS = {
    "21271R": {
        "ms": "./data/demo/21271R/Inscopix_flip_flip.tif",
        "conf": "./data/demo/21271R/ZEISS_405.tif",
        "flip": False,
        "scal_init": 1 / 1.9,
    },
    "21272R": {
        "ms": "./data/demo/21272R/Inscopix_Fip_fip.tif",
        "conf": "./data/demo/21272R/ZEISS_405.tif",
        "flip": False,
        "scal_init": 1 / 1.9,
    },
    "25607": {
        "ms": "./data/demo/25607/Insccopix.tif",
        "conf": "./data/demo/25607/ZEISS_405.tif",
        "flip": True,
        "scal_init": 1,
    },
}
PARAM_VERBOSE = True
FIG_PATH = "./figs/co-registration"

os.makedirs(FIG_PATH, exist_ok=True)


# %% load and coregistration
for dsname, dsdat in DS.items():
    im_ms, im_conf = read_templates(dsdat["ms"], dsdat["conf"], flip=dsdat["flip"])
    im_ms_ps = xr.apply_ufunc(
        process_temp,
        im_ms,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        kwargs={"back_wnd": (61, 61)},
    ).rename("ms-ps")
    im_conf_ps = xr.apply_ufunc(
        process_temp,
        im_conf,
        input_core_dims=[["height", "width"]],
        output_core_dims=[["height", "width"]],
        kwargs={"back_wnd": (151, 151)},
    ).rename("conf-ps")
    tx, param_df = est_sim(
        im_ms_ps.data,
        im_conf_ps.data,
        scal_init=dsdat["scal_init"],
        scal_stp=5e-3,
        scal_nstp=2,
        trans_stp=(3, 3),
        trans_nstp=(15, 15),
        ang_stp=np.deg2rad(3),
        ang_nstp=5,
        verbose=PARAM_VERBOSE,
    )
    im_ms_reg = xr.DataArray(
        apply_tx(im_ms, tx, ref=im_conf.data),
        dims=["height", "width"],
        coords={"height": im_conf.coords["height"], "width": im_conf.coords["width"]},
        name="ms-reg",
    )
    ps_ms_reg = xr.DataArray(
        apply_tx(im_ms_ps, tx, ref=im_conf_ps),
        dims=["height", "width"],
        coords={"height": im_conf.coords["height"], "width": im_conf.coords["width"]},
        name="ps-reg",
    )
    ps_diff = (im_conf_ps - ps_ms_reg).rename("ps-diff")
    im_diff = (im_conf - im_ms_reg).rename("diff")

    fig = plot_ims(
        [im_ms, im_ms_ps, im_conf, im_conf_ps, im_ms_reg, ps_diff, im_diff],
        facet_col_wrap=4,
        norm=True,
    )
    fig.write_html(os.path.join(FIG_PATH, "{}.html".format(dsname)))
