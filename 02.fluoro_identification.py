# %% import and definition
import os

import numpy as np
import pandas as pd
import xarray as xr

from routine.fluoro_id import (
    beta_ztrans,
    classify_units,
    fit_spec,
    max_agg_beta,
    merge_passes,
)
from routine.io import load_dataset, load_refmat

IN_SS_CSV = "./data/full/sessions.csv"
IN_DPATH = "./data/full"
IN_REF_PATH = "./data/ref/Detection_Constants.mat"
IN_SPEC_PATH = "./intermediate/spectrum/"
OUT_PATH = "./output/cell_labs"
PARAM_ZTHRES = 1.5
PARAM_NFLUO = 2
PARAM_EXC_FLUO = []

os.makedirs(OUT_PATH, exist_ok=True)


# %% load data
spec_ref, pdist = load_refmat(IN_REF_PATH)
for (anm, ss), ds, ssrow in load_dataset(
    IN_DPATH, IN_SS_CSV, load_temps=False, load_rois=False, load_specs=False
):
    dsname = "{}-{}".format(anm, ss)
    spec_ds = xr.open_dataset(os.path.join(IN_SPEC_PATH, "{}.nc".format(dsname)))
    spec_raw, spec_norm = spec_ds["spec_raw"].dropna("roi_id"), spec_ds[
        "spec_norm"
    ].dropna("roi_id")
    beta_raw = fit_spec(spec_raw, spec_ref)
    beta_norm = fit_spec(spec_norm, spec_ref)
    beta_raw_z = beta_ztrans(beta_raw)
    beta_norm_z = beta_ztrans(beta_norm)
    thres_raw = (
        PARAM_ZTHRES
        * pdist.sel(dist="raw")
        / np.abs(beta_raw.mean("roi_id") / beta_raw.mean("roi_id").sum())
    ).to_series()
    thres_norm = (
        PARAM_ZTHRES
        * pdist.sel(dist="norm")
        / np.abs(beta_norm.mean("roi_id") / beta_norm.mean("roi_id").sum())
    ).to_series()
    beta_z = max_agg_beta(beta_raw_z, beta_norm_z).to_dataframe().reset_index()
    beta_raw_z = beta_raw_z.rename("beta").to_dataframe().reset_index()
    beta_norm_z = beta_norm_z.rename("beta").to_dataframe().reset_index()
    labs_p1 = classify_units(beta_z, src="p1", zthres=PARAM_ZTHRES)
    labs_p2_raw = classify_units(beta_raw_z, src="p2_raw", zthres=thres_raw)
    labs_p2_norm = classify_units(beta_norm_z, src="p2_norm", zthres=thres_norm)
    labs = pd.concat([labs_p1, labs_p2_raw, labs_p2_norm], ignore_index=True)
    labs = labs[~labs["lab"].isin(PARAM_EXC_FLUO)]
    labs = (
        labs.groupby("roi_id")
        .apply(merge_passes, nfluo=PARAM_NFLUO, include_groups=False)
        .reset_index()
    )
    labs.to_csv(os.path.join(OUT_PATH, "{}.csv".format(dsname)))
