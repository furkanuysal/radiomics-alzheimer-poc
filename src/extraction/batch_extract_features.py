import os
import sys
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# importing dataset_loader module
# (Direct import works because they are in the same folder)
try:
    from dataset_loader import build_dataset_index
except ImportError:
    # Precaution in case of python path issues
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from dataset_loader import build_dataset_index

# Location where the output file will be saved (outputs folder in project root)
OUTPUT_PATH = os.path.join("outputs", "features", "feature_matrix.csv")

def create_extractor():
    settings = {
        # [1,1,1] is safe because OASIS-1 is standardized (T88).
        # [2,2,2] can be done for speed, but T88 might already be low resolution.
        "resampledPixelSpacing": [1, 1, 1], 
        "interpolator": sitk.sitkBSpline,
        "binWidth": 25,
        "normalize": True,       
        "normalizeScale": 100,   
        "voxelArrayShift": 300   
    }

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    extractor.disableAllImageTypes()
    
    # Enabling Wavelet (Very important for texture analysis)
    # If it's too slow, you can remove "Wavelet" from the list.
    for image_type in ["Original", "Wavelet"]: 
        extractor.enableImageTypeByName(image_type)

    extractor.disableAllFeatures()
    feature_classes = [
        "firstorder", "shape", "glcm", 
        "glrlm", "glszm", "gldm", "ngtdm"
    ]
    for feature_class in feature_classes:
        extractor.enableFeatureClassByName(feature_class)

    return extractor

def create_simple_mask(image):
    # Since images are already "masked_gfc" (skull-stripped)
    # simple threshold is enough to remove the background (black).
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    
    if sitk.GetArrayFromImage(mask).sum() == 0:
        mask = sitk.OtsuThreshold(image, 1, 0, 200)

    return mask

def process_single_subject(item):
    """Function processing a single patient (For Multiprocessing)"""
    try:
        subj_id = item["id"]
        image_path = item["image_path"]
        cdr = item["cdr"]

        # Extractor must be recreated within each process
        extractor = create_extractor()

        # Read the image
        image = sitk.ReadImage(image_path)
        
        # Create the mask
        mask = create_simple_mask(image)

        # Extract features
        features = extractor.execute(image, mask)

        # Result dictionary
        feature_row = {
            "id": subj_id,
            "cdr": cdr,
            "mmse": item.get("mmse") # Get MMSE if available
        }

        # Take only numerical features, skip diagnostic (metadata) ones
        for k, v in features.items():
            if not k.startswith("diagnostics_"):
                feature_row[k] = v
                
        return feature_row

    except Exception as e:
        # If an error occurs, print to screen but do not stop the process
        print(f"\n[ERROR] {item['id']}: {e}")
        return None

def main():
    print("Scanning dataset...")
    dataset, _, _, _ = build_dataset_index()
    
    print(f"Total number of patients to process: {len(dataset)}")
    print(f"Processing starting (Multiprocessing)...")
    print("-" * 50)

    rows = []
    
    # Perform parallel processing using cores on your computer
    with ProcessPoolExecutor() as executor:
        # tqdm adds a progress bar
        results = list(tqdm(executor.map(process_single_subject, dataset), total=len(dataset)))

    # Clean up failed ones (those returning None)
    rows = [r for r in results if r is not None]

    if not rows:
        print("ERROR: No features extracted!")
        return

    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("-" * 50)
    print(f"Process Completed! ✅")
    print(f"Successful: {len(df)} / {len(dataset)}")
    print(f"File saved: {OUTPUT_PATH}")
    print(f"Matrix Size: {df.shape}") # (39, ~1000+)

if __name__ == "__main__":
    # This block is mandatory for multiprocessing on Windows
    main()