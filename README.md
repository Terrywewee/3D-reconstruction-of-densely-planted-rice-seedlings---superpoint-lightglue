# 3D-reconstruction-of-densely-planted-rice-seedlings---superpoint-lightglue
# Rice Seedling 3D Reconstruction Pipeline 🌾

A modular pipeline for performing 3D reconstruction of densely planted rice seedlings.  
This project includes two main parts:
- Semantic segmentation of rice seedlings in RGB images.
- - Sparse & dense 3D reconstruction using SfM/MVS.

---
root/
root/
├── segmentation/ # Segmentation model and mask generation
│ ├── run_segment.py
│ └── utils.py
├── reconstruction/ # SfM & MVS pipeline (COLMAP based)
│ ├── match_and_sfm.py
│ └── run_dense.py
├── images/ # Original input images
├── masks/ # Segmentation masks (output)
├── sparse/ # Sparse reconstruction output
├── dense/ # Dense reconstruction output
└── README.md

Step 1 Installation 
This project requires both Python (for segmentation and feature matching) and C++ tools (COLMAP, OpenMVS) for reconstruction.
Make sure COLMAP is installed and accessible.

Step 2 Run Sparse Reconstruction
cd reconstruction/
Place  RGB images into a directory like ./images/.
python main.py
Run  Sparse Reconstruction with SuperPoint + LightGlue matching

Configurations can be edited via YAML files in config/. You can adjust:
Number of keypoints
Match threshold
Model backend (e.g., CPU vs CUDA)
Image resize/crop

This will output:
--images.bin/txt
--cameras.bin/txt
--points3D.bin/txt
--database.db  

Step 3 Segmentation
cd segmentation/
python predict_masks.py \
This will produce binary masks like:
masks/
├── image1_mask.png   # Foreground in white (255), background black (0)
├── image2_mask.png
└── ...

Step 4 Dense reconstruction
OpenMVS is used for dense reconstruction.
Clone and build OpenMVS:
https://github.com/cdcseacave/openMVS

# 1. Convert COLMAP model to OpenMVS format
InterfaceCOLMAP \
  -i model/ \
  -o openmvs/scene.mvs \
  -w images_masked

# 2. Densify the point cloud
DensifyPointCloud \
  openmvs/scene.mvs \
  --resolution-level 1 \
  --min-resolution 640 \
  --number-views 6 \
  --max-threads 8

# 3. (Optional) Reconstruct mesh
ReconstructMesh openmvs/scene_dense.mvs




