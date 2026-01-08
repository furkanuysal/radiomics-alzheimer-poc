import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from dataset_loader import build_dataset_index

OUTPUT_PATH = "outputs/features/feature_matrix.csv"


def create_extractor():
    settings = {
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": sitk.sitkBSpline,
        "binWidth": 25,
        "normalize": True,
        "normalizeScale": 100
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    # Enable image types
    extractor.disableAllImageTypes()
    for image_type in ["Original", "Wavelet"]:
        extractor.enableImageTypeByName(image_type)

    # Enable feature classes
    extractor.disableAllFeatures()
    feature_classes = [
        "firstorder", "shape", "glcm", 
        "glrlm", "glszm", "gldm", "ngtdm"
    ]
    for feature_class in feature_classes:
        extractor.enableFeatureClassByName(feature_class)

    return extractor



def create_simple_mask(image):
    # Simple full-volume mask (for now)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    return mask


def main():
    print("Loading dataset index...")
    dataset, _, _, _ = build_dataset_index()

    print(f"Total subjects to process: {len(dataset)}")

    extractor = create_extractor()

    rows = []

    for i, item in enumerate(dataset):
        subj_id = item["id"]
        image_path = item["image_path"]
        cdr = item["cdr"]

        if pd.isna(cdr):
            print(f"[SKIP] {subj_id} has no CDR label.")
            continue

        print(f"[{i+1}/{len(dataset)}] Processing {subj_id}")

        try:
            image = sitk.ReadImage(image_path)
            mask = create_simple_mask(image)

            features = extractor.execute(image, mask)

            feature_row = {
                "id": subj_id,
                "cdr": cdr
            }

            for k, v in features.items():
                if k.startswith("original"):
                    feature_row[k] = v

            rows.append(feature_row)

        except Exception as e:
            print(f"[ERROR] {subj_id}: {e}")

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nFeature matrix saved to: {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()
