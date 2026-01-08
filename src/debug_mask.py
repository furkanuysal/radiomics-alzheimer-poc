import os
import sys
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np

# Import dataset_loader module
try:
    from dataset_loader import build_dataset_index
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from dataset_loader import build_dataset_index

# Import mask functions from feature_extraction_hippo
from feature_extraction_hippo import get_fseg_path, create_hippocampus_mask

def visualize_mask():
    print("Loading dataset...")
    dataset, _, _, _ = build_dataset_index()
    
    # Let's take the first patient (or a random one)
    sample = dataset[0] 
    
    subj_id = sample["id"]
    image_path = sample["image_path"]
    
    print(f"Examined Patient: {subj_id}")
    print(f"Image Path: {image_path}")

    # Read Image
    image = sitk.ReadImage(image_path)
    image_arr = sitk.GetArrayFromImage(image)

    # Find FSL Path
    fseg_path = get_fseg_path(image_path)
    if not fseg_path:
        print("❌ ERROR: FSL Segmentation file not found!")
        return

    print(f"FSL Path: {fseg_path}")
    fseg_image = sitk.ReadImage(fseg_path)

    # Generate Mask
    print("Generating mask...")
    mask = create_hippocampus_mask(image, fseg_image)
    mask_arr = sitk.GetArrayFromImage(mask)

    # Is Mask Full or Empty?
    voxel_count = np.sum(mask_arr)
    print(f"Voxel Count Inside Mask: {voxel_count}")
    
    if voxel_count == 0:
        print("❌ ERROR: Mask is completely empty! (Coordinates or FSL label wrong)")
    else:
        print("✅ Mask is not empty, visualizing...")

    # Visualization (Middle Section - Axial, Coronal, Sagittal)
    # Find the center where the mask is dense
    if voxel_count > 0:
        z, y, x = np.where(mask_arr == 1)
        center_z, center_y, center_x = int(np.mean(z)), int(np.mean(y)), int(np.mean(x))
    else:
        # If mask is empty, take the center of the image
        center_z, center_y, center_x = np.array(image_arr.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial (Top view)
    axes[0].imshow(image_arr[center_z, :, :], cmap="gray")
    axes[0].imshow(mask_arr[center_z, :, :], cmap="jet", alpha=0.5) # Overlay mask
    axes[0].set_title(f"Axial (Z={center_z})")

    # Coronal (Front view)
    axes[1].imshow(image_arr[:, center_y, :], cmap="gray")
    axes[1].imshow(mask_arr[:, center_y, :], cmap="jet", alpha=0.5)
    axes[1].set_title(f"Coronal (Y={center_y})")

    # Sagittal (Side view)
    axes[2].imshow(image_arr[:, :, center_x], cmap="gray")
    axes[2].imshow(mask_arr[:, :, center_x], cmap="jet", alpha=0.5)
    axes[2].set_title(f"Sagittal (X={center_x})")

    plt.suptitle(f"Hippocampus Mask Check: {subj_id}", fontsize=16)
    
    output_img = "outputs/figures/debug_mask_output.png"
    plt.savefig(output_img)
    print(f"\n📸 Image saved: {output_img}")

if __name__ == "__main__":
    visualize_mask()