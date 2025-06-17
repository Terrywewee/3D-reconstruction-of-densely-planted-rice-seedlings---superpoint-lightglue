
#  3D Reconstruction of Densely Planted Rice Seedlings  
### Rice Seedlings 3D Reconstruction pipeline 🌾

A modular pipeline for performing 3D reconstruction of densely planted rice seedlings.

This project includes two main parts:

-  Semantic segmentation of rice seedlings in RGB images using DeepLabv3+
-  Sparse & dense 3D reconstruction using sfm/mvs

---

## Project Structure

```text
root/
├── segmentation/           # Segmentation model and mask generation
│   ├── run_segment.py
│   └── utils.py
├── reconstruction/         # SfM & MVS pipeline (COLMAP + OpenMVS)
│   ├── match_and_sfm.py
│   └── run_main.py
├── images/                 # Original input images
├── masks/                  # Segmentation masks (output)
├── sparse/                 # Sparse reconstruction output
├── dense/                  # Dense reconstruction output
└── README.md
```

---

## Step 1: Installation

This project requires both Python (for segmentation and matching) and C++ tools (COLMAP, OpenMVS) for reconstruction.

- Install [COLMAP](https://colmap.github.io/)
- Build [OpenMVS](https://github.com/cdcseacave/openMVS)
- Install Python dependencies:
Make sure `colmap`, and other OpenMVS tools are in your system PATH.

---

## Step 2: Sparse Reconstruction

```bash
cd reconstruction/
Place  RGB images into a directory like ./images/.
python main.py
Run  Sparse Reconstruction with SuperPoint + LightGlue matching
```

TConfigurations can be edited via YAML files in config/. You can adjust:
Number of keypoints
Match threshold
Model backend (e.g., CPU vs CUDA)
Image resize/crop


**This will output**:

```text
sparse/
├── images.bin/txt
├── cameras.bin/txt
├── points3D.bin/txt
└── database.db
```

---

## Step 3: Segmentation (Foreground Extraction)

Use DeepLabv3+ to segment rice seedlings from background (soil, tray, etc.)

```bash
cd segmentation/
python predict_masks.py --input_dir ../images --output_dir ../masks --model_path deeplabv3plus_rice.pth
```

This will produce binary masks:

```text
masks/
├── image1_mask.png   # Foreground in white (255), background in black (0)
├── image2_mask.png
└── ...
```

You can then use these masks to generate foreground-only RGB images:

```python
# Example code to apply mask
masked = image * (mask / 255)[:, :, None]
```

---

## Step 4: Dense Reconstruction (OpenMVS)

OpenMVS is used for dense reconstruction.

### 1. Convert COLMAP model to OpenMVS format

```bash
InterfaceCOLMAP   -i sparse/   -o openmvs/scene.mvs   -w images_masked
```

> Make sure `images_masked/` contains the masked RGB images created using segmentation masks.

---

### 2. Densify the point cloud

```bash
DensifyPointCloud   openmvs/scene.mvs   --resolution-level 1   --min-resolution 640   --number-views 6   --max-threads 8
```

---

---

##  Output Example

```text
openmvs/
├── scene.mvs
├── scene_dense.ply
├── scene_dense_mesh.ply

```

You can visualize outputs using MeshLab, Open3D, or CloudCompare.

---


