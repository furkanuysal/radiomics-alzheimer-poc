import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import joblib
from radiomics import featureextractor

MODEL_PATH = "outputs/models/hippo_mmse_regression.pkl"
ROI_BOX = {"x": (55, 75), "y": (105, 135), "z": (45, 65)}

def create_extractor():
    settings = {
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": sitk.sitkBSpline,
        "binWidth": 25,
        "normalize": True,
        "normalizeScale": 100,
        "voxelArrayShift": 300
    }
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.enableImageTypeByName("Wavelet")
    extractor.disableAllFeatures()
    for fc in ["firstorder","shape","glcm","glrlm","glszm","gldm","ngtdm"]:
        extractor.enableFeatureClassByName(fc)
    return extractor

def create_roi_mask(image):
    arr = sitk.GetArrayFromImage(image)
    mask = np.zeros_like(arr, dtype=np.uint8)

    x_min, x_max = ROI_BOX["x"]
    y_min, y_max = ROI_BOX["y"]
    z_min, z_max = ROI_BOX["z"]

    mask[z_min:z_max, y_min:y_max, x_min:x_max] = 1
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(image)
    return mask_img

def extract_features(image_path):
    image = sitk.ReadImage(image_path)
    mask = create_roi_mask(image)

    extractor = create_extractor()
    features = extractor.execute(image, mask)

    filtered = {k:v for k,v in features.items() if not k.startswith("diagnostics_")}
    return pd.DataFrame([filtered])

def interpret_mmse(score):
    if score >= 28:
        return "NORMAL"
    elif score >= 24:
        return "MILD IMPAIRMENT"
    elif score >= 18:
        return "MODERATE ALZHEIMER"
    else:
        return "SEVERE ALZHEIMER"

def main():
    image_path = input("MRI file path: ")

    if not os.path.exists(image_path):
        print("MRI could not be found.")
        return

    print("Radiomics features are being extracted...")

    df = extract_features(image_path)

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_columns"]

    df = df.reindex(columns=feature_cols, fill_value=0)

    pred_mmse = pipeline.predict(df)[0]

    print("\n=== RESULT ===")
    print("Predicted MMSE:", round(pred_mmse, 2))
    print("Cognitive Level:", interpret_mmse(pred_mmse))
    print("Note: This value may vary by ± 2.9 points.")
    print("Note: This value shouldn't be used as a clinical diagnosis.")
    
if __name__ == "__main__":
    main()
