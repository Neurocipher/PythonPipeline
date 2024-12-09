# %% imports
import itertools as itt
import os
import pickle as pkl

import numpy as np
import xarray as xr
import yaml
from pydantic.v1.utils import deep_update

from routine.coregistration import apply_tx, estimate_tranform, process_temp
from routine.io import load_dataset
from routine.plotting import plot_ims

IN_DPATH = "./data/full/"
IN_SS_CSV = "./data/full/sessions.csv"
IN_PARAM_PATH = "./params/"
SKIP_EXISTING = False
OUT_PATH = "./intermediate/co-registration"
FIG_PATH = "./figs/co-registration"

os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)


# %% load and coregistration
for (anm, ss), ds, ssrow in load_dataset(
    IN_DPATH, IN_SS_CSV, load_rois=False, load_specs=False
):
    im_ms, im_conf = ds["im_ms"], ds["im_conf"]
    dsname = "{}-{}".format(anm, ss)
    if SKIP_EXISTING and os.path.exists(os.path.join(OUT_PATH, "{}.nc".format(dsname))):
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
