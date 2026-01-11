import os
import sys
import pandas as pd
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

try:
    from dataset_loader import build_dataset_index
except ImportError:
    current_file_path = os.path.abspath(__file__)
    extraction_dir = os.path.dirname(current_file_path)
    src_dir = os.path.dirname(extraction_dir)
    utils_dir = os.path.join(src_dir, "utils")

    if utils_dir not in sys.path:
        sys.path.append(utils_dir)
    from dataset_loader import build_dataset_index

OUTPUT_PATH = "outputs/features/hippocampus/feature_matrix_hippo.csv"

# --- UPDATED "TIGHT" COORDINATES (T88 Atlas) ---

# Right Hippocampus (Image Left)
# Adjusted X from 45 to 55 (Tightened from outside in)
# Reduced Z from 40-80 to 45-65 (Reduced height)
RIGHT_HIPPO_BOX = {
    "x": (55, 75),   
    "y": (105, 135), 
    "z": (45, 65)    
}

# Left Hippocampus (Image Right)
# Adjusted X from 130 to 120 (Tightened from outside in)
LEFT_HIPPO_BOX = {
    "x": (100, 120), 
    "y": (105, 135), 
    "z": (45, 65)    
}

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
    
    # Shape features are now much more important!
    for image_type in ["Original", "Wavelet"]: 
        extractor.enableImageTypeByName(image_type)
    
    extractor.disableAllFeatures()
    feature_classes = ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
    for fc in feature_classes:
        extractor.enableFeatureClassByName(fc)
    
    return extractor

def get_fseg_path(image_path):
    dirname = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    fseg_name = filename.replace("_masked_gfc.img", "_masked_gfc_fseg.img")
    
    parts = image_path.split(os.sep)
    try:
        proc_idx = parts.index("PROCESSED")
        subject_root = os.sep.join(parts[:proc_idx])
        fseg_path = os.path.join(subject_root, "FSL_SEG", fseg_name)
        if os.path.exists(fseg_path):
            return fseg_path
        return None
    except ValueError:
        return None

def create_hippocampus_mask(image, fseg_image):
    """
    FSL Segmentation (ONLY GRAY MATTER) + Narrowed Double Box
    """
    fseg_arr = sitk.GetArrayFromImage(fseg_image)
    
    # --- CORRECTION HERE ---
    # Label 3 (White Matter) REMOVED. Only taking Label 2 (Gray Matter).
    # This will reduce mask size by 70% and focus only on thinking cells.
    tissue_mask = (fseg_arr == 2).astype(np.uint8)
    
    roi_mask = np.zeros_like(tissue_mask)
    
    # --- ADD RIGHT BOX ---
    x_min, x_max = RIGHT_HIPPO_BOX["x"]
    y_min, y_max = RIGHT_HIPPO_BOX["y"]
    z_min, z_max = RIGHT_HIPPO_BOX["z"]
    roi_mask[z_min:z_max, y_min:y_max, x_min:x_max] = 1
    
    # --- ADD LEFT BOX ---
    x_min, x_max = LEFT_HIPPO_BOX["x"]
    y_min, y_max = LEFT_HIPPO_BOX["y"]
    z_min, z_max = LEFT_HIPPO_BOX["z"]
    roi_mask[z_min:z_max, y_min:y_max, x_min:x_max] = 1
    
    # INTERSECTION
    final_mask_arr = tissue_mask * roi_mask
    
    # Warn if too small (but continue)
    if final_mask_arr.sum() < 100:
        pass 

    final_mask = sitk.GetImageFromArray(final_mask_arr)
    final_mask.CopyInformation(image)
    
    return final_mask

def process_single_subject(item):
    try:
        subj_id = item["id"]
        image_path = item["image_path"]
        
        image = sitk.ReadImage(image_path)
        fseg_path = get_fseg_path(image_path)
        
        if not fseg_path:
            return None
            
        fseg_image = sitk.ReadImage(fseg_path)
        mask = create_hippocampus_mask(image, fseg_image)
        
        # Check if mask is empty as radiomics will error
        if sitk.GetArrayFromImage(mask).sum() == 0:
            return None

        # Radiomics Analysis (Label=1)
        extractor = create_extractor()
        features = extractor.execute(image, mask, label=1)

        feature_row = {"id": subj_id, "cdr": item["cdr"]}
        for k, v in features.items():
            if not k.startswith("diagnostics_"):
                feature_row[k] = v
                
        return feature_row

    except Exception as e:
        # Disabled error printing to keep screen clean, just return None
        return None

def main():
    print("Scanning dataset...")
    dataset, _, _, _ = build_dataset_index()
    print(f"Total: {len(dataset)} patients. Hippocampus feature extraction starting...")
    
    rows = []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_subject, dataset), total=len(dataset)))

    rows = [r for r in results if r is not None]
    
    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        print("-" * 30)
        print(f"Done! New matrix: {df.shape}")
    else:
        print("Error: No data could be extracted.")

if __name__ == "__main__":
    main()