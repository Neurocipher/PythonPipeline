import re

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk

from .utilities import normalize


def process_temp(
    im: np.ndarray, med_wnd=5, back_wnd=(101, 101), blk_wnd=(11, 11), q_thres=None
):
    im_ps = cv2.medianBlur(im.astype(np.float32), med_wnd).astype(float)
    back = cv2.blur(im_ps, back_wnd)
    im_ps = im_ps - back
    krn = cv2.getStructuringElement(cv2.MORPH_RECT, blk_wnd)
    im_ps = cv2.morphologyEx(im_ps, cv2.MORPH_BLACKHAT, krn)
    if q_thres is not None:
        q = np.quantile(im_ps, q_thres)
        im_ps[im_ps < q] = im_ps.min()
    return normalize(im_ps)


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
    src_ma=None,
    dst_ma=None,
    lr: float = 0.5,
    niter: int = 1000,
    scal_init=None,
    scal_stp=1e-2,
    scal_nstp=3,
    ang_stp=np.deg2rad(1),
    ang_nstp=3,
    trans_stp=(1.0, 1.0),
    trans_nstp=(3, 3),
    verbose=False,
):
    src = sitk.GetImageFromArray(src.astype(np.float32))
    dst = sitk.GetImageFromArray(dst.astype(np.float32))
    reg = sitk.ImageRegistrationMethod()
    if src_ma is not None:
        reg.SetMetricMovingMask(sitk.GetImageFromArray(src_ma.astype(np.uint8)))
    if dst_ma is not None:
        reg.SetMetricFixedMask(sitk.GetImageFromArray(dst_ma.astype(np.uint8)))
    if scal_init is None:
        scal_init = 1 / 2.5
    trans_opt = sitk.CenteredTransformInitializer(
        dst,
        src,
        sitk.Similarity2DTransform(scal_init),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    # reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    # reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    reg.SetInitialTransform(trans_opt)
    # reg.SetMetricAsMeanSquares()
    # reg.SetMetricAsANTSNeighborhoodCorrelation(2)
    reg.SetMetricAsCorrelation()
    reg.SetInterpolator(sitk.sitkLinear)
    # reg.SetOptimizerAsRegularStepGradientDescent(
    #     learningRate=lr,
    #     minStep=1e-7,
    #     numberOfIterations=niter,
    # )
    reg.SetOptimizerAsExhaustive([scal_nstp, ang_nstp, trans_nstp[0], trans_nstp[1]])
    reg.SetOptimizerScales([scal_stp, ang_stp, trans_stp[0], trans_stp[1]])
    # reg.SetOptimizerScalesFromPhysicalShift()
    param_dict = dict()
    reg.AddCommand(sitk.sitkIterationEvent, lambda: it_callback(reg, param_dict))
    tx = reg.Execute(dst, src).Downcast()
    param_df = (
        pd.Series(param_dict)
        .reset_index(name="metric")
        .rename(lambda c: re.sub("level_", "param_", c), axis="columns")
    )
    if verbose:
        print("Scales: {}".format(param_df["param_0"].unique()))
        print("Angles: {}".format(param_df["param_1"].unique()))
        print("TransX: {}".format(param_df["param_2"].unique()))
        print("TransY: {}".format(param_df["param_3"].unique()))
        print(
            "Scale: {}, Angle: {}, Trans: {}".format(
                tx.GetScale(), tx.GetAngle(), tx.GetTranslation()
            )
        )
    return tx, param_df
