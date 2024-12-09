# %% imports
import itertools as itt
import os
import pickle as pkl

import numpy as np
import xarray as xr

from routine.coregistration import apply_tx, estimate_tranform, process_temp
from routine.io import read_templates
from routine.plotting import plot_ims

DS = {
    "21271R": {
        "ms": "./data/demo/21271R/Inscopix_flip_flip.tif",
        "conf": "./data/demo/21271R/ZEISS_405.tif",
        "flip": False,
        "scal_init": 1.9,
    },
    "21272R": {
        "ms": "./data/demo/21272R/Inscopix_Fip_fip.tif",
        "conf": "./data/demo/21272R/ZEISS_405.tif",
        "flip": False,
        "scal_init": 1.9,
    },
    "25607": {
        "ms": "./data/demo/25607/Insccopix.tif",
        "conf": "./data/demo/25607/ZEISS_405.tif",
        "flip": True,
        "scal_init": 1,
    },
}
OUT_PATH = "./intermediate/co-registration"
FIG_PATH = "./figs/co-registration"

os.makedirs(OUT_PATH, exist_ok=True)
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
    tx, tx_exh, param_df = estimate_tranform(
        im_ms_ps.data,
        im_conf_ps.data,
        scal_init=dsdat["scal_init"],
        scal_stp=5e-2,
        scal_nstp=2,
        trans_stp=(5, 5),
        trans_nstp=(12, 12),
        ang_stp=np.deg2rad(5),
        ang_nstp=3,
        lr=1e-3,
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
    fig.write_html(os.path.join(FIG_PATH, "{}.html".format(dsname)))
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
    ds.to_netcdf(os.path.join(OUT_PATH, "{}.nc".format(dsname)))
    with open(os.path.join(OUT_PATH, "tx-{}.pkl".format(dsname)), "wb") as tx_file:
        pkl.dump(tx, tx_file)
    print(
        "data: {}, scale: {}, angle: {}, shift: {}".format(
            dsname, 1 / tx.GetScale(), np.rad2deg(tx.GetAngle()), tx.GetTranslation()
        )
    )

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
