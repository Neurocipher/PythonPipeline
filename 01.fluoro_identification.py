# %% import and definition
import os

from routine.fluoro_id import beta_ztrans, classify_unit, fit_spec, max_agg_beta
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
    beta_z = max_agg_beta(beta_raw_z, beta_norm_z).to_dataframe().reset_index()
    labs = (
        beta_z.groupby("unit")
        .apply(
            classify_unit,
            zthres=PARAM_ZTHRES,
            nfluo=PARAM_NFLUO,
            exc_lab=PARAM_EXC_FLUO,
            include_groups=False,
        )
        .reset_index()
    )
    labs.to_csv(os.path.join(OUT_PATH, "{}.csv".format(cell_mat.split("_")[0])))
