import os
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor


def load_image(image_path):
    """Load MRI image using SimpleITK."""
    image = sitk.ReadImage(image_path)
    return image


def create_simple_mask(image, threshold=0):
    """
    Create a simple binary mask covering the brain region.
    This is a placeholder ROI for pipeline testing.
    """
    image_array = sitk.GetArrayFromImage(image)
    mask_array = (image_array > threshold).astype("uint8")

    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(image)
    return mask


def extract_radiomics(image, mask, params=None):
    """
    Extract radiomics features using PyRadiomics.
    """
    if params is None:
        extractor = featureextractor.RadiomicsFeatureExtractor()
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor(**params)

    result = extractor.execute(image, mask)
    return result


def main():
    # --- Paths ---
    image_path = os.path.join(
        "data", "raw", "disc1", "OAS1_0001_MR1", "RAW",
        "OAS1_0001_MR1_mpr-1_anon.hdr"
    )

    output_dir = os.path.join("outputs", "features")
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "features_baseline.csv")

    # --- Load image ---
    print("Loading image...")
    image = load_image(image_path)

    # --- Create ROI mask ---
    print("Creating simple ROI mask...")
    mask = create_simple_mask(image)

    # --- Extract radiomics ---
    print("Extracting radiomics features...")
    features = extract_radiomics(image, mask)

    # --- Convert to DataFrame ---
    df = pd.DataFrame.from_dict(features, orient="index", columns=["value"])
    df.reset_index(inplace=True)
    df.columns = ["feature", "value"]

    # --- Save ---
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

    # --- Radiomics with resampling ---
    print("Extracting radiomics features with resampling...")

    resampling_params = {
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": sitk.sitkBSpline,
        "binWidth": 25
    }

    features_resampled = extract_radiomics(image, mask, resampling_params)

    df_resampled = pd.DataFrame.from_dict(
        features_resampled, orient="index", columns=["value"]
    )
    df_resampled.reset_index(inplace=True)
    df_resampled.columns = ["feature", "value"]

    output_csv_resampled = os.path.join(
        output_dir, "features_resampled_1mm.csv"
    )

    df_resampled.to_csv(output_csv_resampled, index=False)
    print(f"Resampled features saved to {output_csv_resampled}")
    
        # --- Radiomics with resampling + binWidth = 50 ---
    print("Extracting radiomics features with resampling (binWidth=50)...")

    resampling_bin50_params = {
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": sitk.sitkBSpline,
        "binWidth": 50
    }

    features_bin50 = extract_radiomics(image, mask, resampling_bin50_params)

    df_bin50 = pd.DataFrame.from_dict(
        features_bin50, orient="index", columns=["value"]
    )
    df_bin50.reset_index(inplace=True)
    df_bin50.columns = ["feature", "value"]

    output_csv_bin50 = os.path.join(
        output_dir, "features_resampled_bin50.csv"
    )

    df_bin50.to_csv(output_csv_bin50, index=False)
    print(f"Resampled (binWidth=50) features saved to {output_csv_bin50}")

if __name__ == "__main__":
    main()
