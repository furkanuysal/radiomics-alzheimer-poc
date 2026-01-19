import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import joblib
from radiomics import featureextractor

MODEL_PATH = "outputs/models/hippo_right_texture_model.pkl"

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

def main():
    image_path = input("MRI file path: ")

    if not os.path.exists(image_path):
        print("MRI file not found.")
        return

    print("Radiomics features are being extracted...")

    df = extract_features(image_path)
    print("Feature vector shape:", df.shape)

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    scaler = bundle["scaler"]
    selector = bundle["selector"]
    feature_names = bundle["features"]

    df = df[feature_names]

    X_scaled = scaler.transform(df)
    X_selected = selector.transform(X_scaled)

    pred = model.predict(X_selected)[0]

    print("\n=== RESULT ===")
    print("Prediction:", "ALZHEIMER" if pred == 1 else "HEALTHY")

if __name__ == "__main__":
    main()
