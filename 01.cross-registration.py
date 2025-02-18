# %% import and definition
import os
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from routine.crossreg import (
    agg_cent,
    calculate_centroid_distance,
    calculate_mapping,
    centroid,
    fill_mapping,
    group_by_session,
)
from routine.io import load_dataset

IN_DPATH = "./intermediate/co-registration/rois/"
IN_SS_CSV = "./data/full/sessions.csv"
OUT_PATH = "./intermediate/cross-registration"
FIG_PATH = "./figs/cross-registration"
PARAM_SKIP_EXISTING = False
PARAM_CENT_DIST = 10

os.makedirs(OUT_PATH, exist_ok=True)

# %% compute centroids
if PARAM_SKIP_EXISTING and os.path.exists(os.path.join(OUT_PATH, "cents.feat")):
    pass
else:
    cents = []
    for (anm, ss), ssrow in load_dataset(IN_SS_CSV):
        # load data
        dsname = "{}-{}".format(anm, ss)
        rois = xr.open_dataset(os.path.join(IN_DPATH, "{}.nc".format(dsname)))["rois"]
        cent = centroid(rois)
        cent["animal"] = anm
        cent["session"] = ss
        cents.append(cent)
    cents = pd.concat(cents, ignore_index=True)
    cents.to_feather(os.path.join(OUT_PATH, "cents.feat"))

# %% registration
if PARAM_SKIP_EXISTING and os.path.exists(os.path.join(OUT_PATH, "mapping.feat")):
    pass
else:
    cents = (
        pd.read_feather(os.path.join(OUT_PATH, "cents.feat"))
        .rename(columns={"roi_id": "unit_id"})
        .sort_values(["animal", "session", "unit_id"])
        .set_index(["animal", "session", "unit_id"])
    )
    mappings = []
    for anm, anm_df in cents.groupby("animal"):
        anm_df = anm_df.reset_index()
        roi_ct = (
            anm_df.groupby("session")
            .size()
            .rename("ct")
            .reset_index()
            .sort_values("ct", ascending=False)
        )
        assert len(roi_ct) > 1, "Must have more than two sessions to register"
        cent_base = anm_df[anm_df["session"] == roi_ct["session"].iloc[0]]
        mapping = None
        for ss_new in roi_ct["session"].iloc[1:]:
            cent_new = anm_df[anm_df["session"] == ss_new]
            cent_reg = pd.concat([cent_base, cent_new], ignore_index=True)
            dist = calculate_centroid_distance(cent_reg)
            dist = dist[dist["variable", "distance"] < PARAM_CENT_DIST].copy()
            if len(dist) > 0:
                mapping_new = calculate_mapping(group_by_session(dist))
                if mapping is None:
                    mapping = mapping_new
                else:
                    u_base, u_new = np.array(mapping_new["session", "base"]), np.array(
                        mapping_new["session", ss_new]
                    )
                    mapping.loc[u_base, ("session", ss_new)] = u_new
                    mapping = group_by_session(mapping)
                    mapping = mapping.reindex(sorted(mapping.columns), axis="columns")
            else:
                mapping["session", ss_new] = np.nan
            mapping = fill_mapping(mapping, anm_df)
            cent_base = mapping.apply(agg_cent, axis="columns", cents=cents)
            cent_base["animal"] = anm
            cent_base["session"] = "base"
            cent_base["unit_id"] = cent_base.index
        assert mapping["session"].notnull().sum().sum() == len(anm_df)
        mapping.loc[:, ("meta", "master_id")] = np.arange(len(mapping))
        mappings.append(mapping)
    mappings = pd.concat(mappings, ignore_index=True)
    mappings = mappings.reindex(sorted(mappings.columns), axis="columns")
    mappings.to_feather(os.path.join(OUT_PATH, "mapping.feat"))

# %% generate master footprints
outpath = os.path.join(OUT_PATH, "rois")
os.makedirs(outpath, exist_ok=True)
mapping = pd.read_feather(os.path.join(OUT_PATH, "mapping.feat"))
for anm, anm_df in mapping.groupby(("meta", "animal")):
    if PARAM_SKIP_EXISTING and os.path.exists(
        os.path.join(outpath, "{}.nc".format(anm))
    ):
        continue
    anm_df = anm_df.dropna(axis="columns", how="all").set_index(("meta", "master_id"))
    rois = {
        ss: xr.open_dataset(os.path.join(IN_DPATH, "{}-{}.nc".format(anm, ss)))["rois"]
        for ss in anm_df["session"].columns
    }
    master_rois = []
    for mid, mp in anm_df.iterrows():
        ss = mp["session"].dropna()
        cur_rois = [rois[s].sel(roi_id=u) for s, u in zip(ss.index, ss)]
        mroi = xr.DataArray(
            np.stack(cur_rois, axis=0).mean(axis=0),
            dims=cur_rois[0].dims,
            coords=cur_rois[0].coords,
        ).assign_coords(roi_id=mid, animal=anm)
        master_rois.append(mroi)
    master_rois = xr.concat(master_rois, "roi_id")
    master_rois.rename("rois").to_dataset().to_netcdf(
        os.path.join(outpath, "{}.nc".format(anm))
    )
