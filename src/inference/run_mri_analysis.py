import os
import sys
import json
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk
import joblib
from radiomics import featureextractor


# -----------------------------
# CONFIG
# -----------------------------
DIAG_MODEL_PATH = "outputs/models/hippo_right_texture_model.pkl"
MMSE_MODEL_PATH = "outputs/models/hippo_mmse_regression.pkl"

# Right hippocampus box
ROI_BOX = {"x": (55, 75), "y": (105, 135), "z": (45, 65)}

# MMSE category mapping
def interpret_mmse(score: float) -> str:
    if score >= 28:
        return "NORMAL"
    elif score >= 24:
        return "MILD IMPAIRMENT"
    elif score >= 18:
        return "MODERATE ALZHEIMER"
    else:
        return "SEVERE ALZHEIMER"


# -----------------------------
# Radiomics extractor
# -----------------------------
def create_extractor() -> featureextractor.RadiomicsFeatureExtractor:
    settings = {
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": sitk.sitkBSpline,
        "binWidth": 25,
        "normalize": True,
        "normalizeScale": 100,
        "voxelArrayShift": 300,
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("Wavelet")

    extractor.disableAllFeatures()
    for fc in ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]:
        extractor.enableFeatureClassByName(fc)

    return extractor


def create_roi_mask(image: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image)
    mask = np.zeros_like(arr, dtype=np.uint8)

    x_min, x_max = ROI_BOX["x"]
    y_min, y_max = ROI_BOX["y"]
    z_min, z_max = ROI_BOX["z"]

    mask[z_min:z_max, y_min:y_max, x_min:x_max] = 1

    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(image)
    return mask_img


def extract_features_df(image_path: str) -> pd.DataFrame:
    # Reduce noisy warnings from ITK about Analyze format; keep errors visible
    sitk.ProcessObject_SetGlobalWarningDisplay(False)

    image = sitk.ReadImage(image_path)
    mask = create_roi_mask(image)

    extractor = create_extractor()
    features = extractor.execute(image, mask)

    filtered = {k: v for k, v in features.items() if not k.startswith("diagnostics_")}
    return pd.DataFrame([filtered])


# -----------------------------
# Model runners
# -----------------------------
def run_diagnosis(df_all_features: pd.DataFrame) -> dict:
    """
    Diagnosis model bundle keys (as you observed):
      dict_keys(['scaler', 'selector', 'model', 'features'])
    """
    bundle = joblib.load(DIAG_MODEL_PATH)

    required = ["scaler", "selector", "model", "features"]
    for k in required:
        if k not in bundle:
            raise KeyError(f"Diagnosis model bundle missing key: '{k}'")

    feature_cols = bundle["features"]
    df = df_all_features.reindex(columns=feature_cols, fill_value=0)

    scaler = bundle["scaler"]
    selector = bundle["selector"]
    model = bundle["model"]

    X = scaler.transform(df.values)
    X_sel = selector.transform(X)

    pred = int(model.predict(X_sel)[0])
    label = "ALZHEIMER" if pred == 1 else "HEALTHY"

    # Per your request: do NOT output confidence/probability
    return {"prediction": label}


def run_mmse(df_all_features: pd.DataFrame) -> dict:
    """
    MMSE bundle keys (as in your current inference script):
      bundle["pipeline"], bundle["feature_columns"]
    """
    bundle = joblib.load(MMSE_MODEL_PATH)

    required = ["pipeline", "feature_columns"]
    for k in required:
        if k not in bundle:
            raise KeyError(f"MMSE model bundle missing key: '{k}'")

    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_columns"]

    df = df_all_features.reindex(columns=feature_cols, fill_value=0)

    pred = float(pipeline.predict(df)[0])
    return {
        "predicted": round(pred, 2),
        "cognitive_level": interpret_mmse(pred),
        "error_note": "This value may vary by ± 2.9 points.",
    }


# -----------------------------
# JSON output helpers
# -----------------------------
def json_ok(image_path: str, diag: dict, mmse: dict) -> dict:
    return {
        "status": "ok",
        "input": {
            "path": image_path,
            "file": os.path.basename(image_path),
        },
        "results": {
            "diagnosis": diag,
            "mmse": mmse,
        },
        "disclaimer": "This is not a clinical diagnosis.",
    }


def json_error(image_path: str | None, code: str, message: str) -> dict:
    return {
        "status": "error",
        "error": {
            "code": code,
            "message": message,
        },
        "input": {"path": image_path} if image_path else None,
    }


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    # Quiet down radiomics informational logging (e.g., GLCM symmetry notes)
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    logging.getLogger("radiomics.featureextractor").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*Analyze file and it's deprecated.*")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mri",
        required=True,
        help="Full path to MRI file (e.g., ..._masked_gfc.img)",
    )
    args = parser.parse_args()

    image_path = args.mri.strip().strip('"').strip("'")

    if not image_path:
        print(json.dumps(json_error(None, "EMPTY_PATH", "MRI path is empty."), ensure_ascii=False))
        return 2

    if not os.path.exists(image_path):
        print(json.dumps(json_error(image_path, "NOT_FOUND", "MRI file does not exist."), ensure_ascii=False))
        return 2

    # Basic extension check (to help UX)
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in [".img", ".nii", ".gz", ".nii.gz"]:
        pass

    try:
        df_all = extract_features_df(image_path)

        diag = run_diagnosis(df_all)
        mmse = run_mmse(df_all)

        print(json.dumps(json_ok(image_path, diag, mmse), ensure_ascii=False))
        return 0

    except Exception as e:
        print(json.dumps(json_error(image_path, "RUNTIME_ERROR", str(e)), ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
