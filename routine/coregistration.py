import re
from collections.abc import Iterable

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

from .utilities import normalize


def process_temp(
    im: np.ndarray, dn_sigma=5, back_sigma=50, blk_wnd=(11, 11), q_thres=None
):
    if not isinstance(blk_wnd, Iterable):
        blk_wnd = (blk_wnd, blk_wnd)
    im_ps = gaussian_filter(im, dn_sigma)
    im_ps = remove_background(im_ps, back_sigma)
    krn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, blk_wnd)
    im_ps = cv2.morphologyEx(im_ps, cv2.MORPH_BLACKHAT, krn)
    if q_thres is not None:
        q = np.quantile(im_ps, q_thres)
        im_ps[im_ps < q] = im_ps.min()
    return normalize(im_ps)


def remove_background(im, back_sigma):
    back = gaussian_filter(im, back_sigma)
    return im - back


def apply_tx(
    fm: np.ndarray, tx: sitk.Transform, fill: float = 0, ref: np.ndarray = None
):
    if ref is None:
        ref = fm
    else:
        ref = sitk.GetImageFromArray(ref)
    fm = sitk.GetImageFromArray(fm)
    fm = sitk.Resample(fm, ref, tx, sitk.sitkLinear, fill)
    return sitk.GetArrayFromImage(fm)


def it_callback(reg, param_dict):
    param = reg.GetOptimizerPosition()
    param_dict[param] = reg.GetMetricValue()


def est_sim(
    src: np.ndarray,
    dst: np.ndarray,
    exhaustive: bool,
    trans_init=None,
    src_ma=None,
    dst_ma=None,
    lr: float = 0.5,
    niter: int = 1000,
    scal_init=1.9,
    scal_stp=1e-2,
    scal_nstp=3,
    ang_stp=np.deg2rad(1),
    ang_nstp=3,
    trans_stp=(1.0, 1.0),
    trans_nstp=(3, 3),
):
    src = sitk.GetImageFromArray(src.astype(np.float32))
    dst = sitk.GetImageFromArray(dst.astype(np.float32))
    reg = sitk.ImageRegistrationMethod()
    if src_ma is not None:
        reg.SetMetricMovingMask(sitk.GetImageFromArray(src_ma.astype(np.uint8)))
    if dst_ma is not None:
        reg.SetMetricFixedMask(sitk.GetImageFromArray(dst_ma.astype(np.uint8)))
    if trans_init is None:
        trans_init = sitk.CenteredTransformInitializer(
            dst,
            src,
            sitk.Similarity2DTransform(1 / scal_init),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
    reg.SetInitialTransform(trans_init)
    reg.SetMetricAsCorrelation()
    reg.SetInterpolator(sitk.sitkLinear)
    # reg.SetOptimizerAsRegularStepGradientDescent(
    #     learningRate=lr,
    #     minStep=1e-7,
    #     numberOfIterations=niter,
    # )
    if exhaustive:
        if not isinstance(trans_stp, Iterable):
            trans_stp = (trans_stp, trans_stp)
        if not isinstance(trans_nstp, Iterable):
            trans_nstp = (trans_nstp, trans_nstp)
        reg.SetOptimizerAsExhaustive(
            [scal_nstp, ang_nstp, trans_nstp[0], trans_nstp[1]]
        )
        reg.SetOptimizerScales(
            [
                1 / scal_init - 1 / (scal_init + scal_stp),
                ang_stp,
                trans_stp[0],
                trans_stp[1],
            ]
        )
    else:
        reg.SetOptimizerAsGradientDescent(learningRate=lr, numberOfIterations=niter)
    # reg.SetOptimizerScalesFromPhysicalShift()
    param_dict = dict()
    reg.AddCommand(sitk.sitkIterationEvent, lambda: it_callback(reg, param_dict))
    tx = reg.Execute(dst, src).Downcast()
    param_df = (
        pd.Series(param_dict)
        .reset_index(name="metric")
        .rename(
            columns={
                "level_0": "scale",
                "level_1": "angle",
                "level_2": "transX",
                "level_3": "transY",
            }
        )
    )
    return tx, param_df


def estimate_tranform(src, dst, **kwargs):
    tx_exh, param_exh = est_sim(src, dst, exhaustive=True, **kwargs)
    tx_gd, param_gd = est_sim(src, dst, exhaustive=False, trans_init=tx_exh, **kwargs)
    param_exh["stage"] = "exhaustive"
    param_gd["stage"] = "gradient"
    return tx_gd, tx_exh, pd.concat([param_exh, param_gd], ignore_index=True)
