import os
import sys
import pandas as pd
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial

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

# --- CONFIGURATIONS ---
ROI_CONFIGS = {
    "left": {
        "box": {"x": (100, 120), "y": (105, 135), "z": (45, 65)},
        "output_path": "outputs/features/hippocampus/feature_matrix_hippo_left.csv"
    },
    "right": {
        "box": {"x": (55, 75), "y": (105, 135), "z": (45, 65)}, 
        "output_path": "outputs/features/hippocampus/feature_matrix_hippo_right.csv"
    }
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

# Box coordinates are now parameters
def create_hippocampus_mask(image, fseg_image, box_coords):
    fseg_arr = sitk.GetArrayFromImage(fseg_image)
    tissue_mask = (fseg_arr == 2).astype(np.uint8)
    
    roi_mask = np.zeros_like(tissue_mask)
    
    # Dynamic coordinate usage
    x_min, x_max = box_coords["x"]
    y_min, y_max = box_coords["y"]
    z_min, z_max = box_coords["z"]
    
    roi_mask[z_min:z_max, y_min:y_max, x_min:x_max] = 1

    final_mask_arr = tissue_mask * roi_mask

    final_mask = sitk.GetImageFromArray(final_mask_arr)
    final_mask.CopyInformation(image)

    return final_mask

def process_single_subject(item, box_coords):
    try:
        subj_id = item["id"]
        image_path = item["image_path"]
        
        image = sitk.ReadImage(image_path)
        fseg_path = get_fseg_path(image_path)
        
        if not fseg_path:
            return None
            
        fseg_image = sitk.ReadImage(fseg_path)
        
        # Pass coordinates to mask function
        mask = create_hippocampus_mask(image, fseg_image, box_coords)
        
        if sitk.GetArrayFromImage(mask).sum() == 0:
            return None

        extractor = create_extractor()
        features = extractor.execute(image, mask, label=1)

        feature_row = {
            "id": subj_id,
            "cdr": item["cdr"],
            "mmse": item.get("mmse", np.nan)
        }

        for k, v in features.items():
            if not k.startswith("diagnostics_"):
                feature_row[k] = v
                
        return feature_row

    except Exception as e:
        return None

def main():
    print("Scanning dataset...")
    dataset, _, _, _ = build_dataset_index()
    print(f"Total: {len(dataset)} patients found.")
    
    for side, config in ROI_CONFIGS.items():
        print("=" * 40)
        print(f"Starting processing for: {side.upper()} HIPPOCAMPUS")
        print(f"Box Coords: {config['box']}")
        print("=" * 40)

        rows = []
        
        process_func = partial(process_single_subject, box_coords=config["box"])

        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_func, dataset), total=len(dataset), desc=f"Extracting {side}"))

        rows = [r for r in results if r is not None]
        
        if rows:
            df = pd.DataFrame(rows)
            output_path = config["output_path"]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print("-" * 30)
            print(f"Done ({side})! Saved to: {output_path}")
            print(f"Matrix Shape: {df.shape}")
        else:
            print(f"Error ({side}): No data could be extracted.")

if __name__ == "__main__":
    main()