# %% import and definition
import os

import numpy as np
import pandas as pd

from routine.fluoro_id import (
    beta_ztrans,
    classify_units,
    fit_spec,
    max_agg_beta,
    merge_passes,
)
from routine.io import load_cellsmat, load_refmat

IN_REF_PATH = "./data/ref/Detection_Constants.mat"
IN_DPATH = "./data/cells/"
OUT_PATH = "./output/cell_labs"
PARAM_ZTHRES = 1.5
PARAM_NFLUO = 2
PARAM_EXC_FLUO = []

os.makedirs(OUT_PATH, exist_ok=True)


# %% load data
spec_ref, pdist = load_refmat(IN_REF_PATH)
for cell_mat in filter(lambda fn: fn.endswith(".mat"), os.listdir(IN_DPATH)):
    spec_raw, spec_norm = load_cellsmat(os.path.join(IN_DPATH, cell_mat))
    beta_raw = fit_spec(spec_raw, spec_ref)
    beta_norm = fit_spec(spec_norm, spec_ref)
    beta_raw_z = beta_ztrans(beta_raw)
    beta_norm_z = beta_ztrans(beta_norm)
    thres_raw = (
        PARAM_ZTHRES
        * pdist.sel(dist="raw")
        / np.abs(beta_raw.mean("unit") / beta_raw.mean("unit").sum())
    ).to_series()
    thres_norm = (
        PARAM_ZTHRES
        * pdist.sel(dist="norm")
        / np.abs(beta_norm.mean("unit") / beta_norm.mean("unit").sum())
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
        labs.groupby("unit")
        .apply(merge_passes, nfluo=PARAM_NFLUO, include_groups=False)
        .reset_index()
    )
    labs.to_csv(os.path.join(OUT_PATH, "{}.csv".format(cell_mat.split("_")[0])))
