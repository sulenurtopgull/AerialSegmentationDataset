# Dataset Card: Aerial Segmentation Dataset

## Task

Semantic segmentation of aerial imagery for multi-class land-cover mapping.

## Data Format

- Input images: RGB aerial image patches in `.tif` format.
- Labels: pixel-level segmentation masks in `.tif` format.
- Folder structure: `train`, `val`, and `test` splits with paired `images` and `masks` directories.

## Dataset Split

| Split | Images | Masks |
|:-----:|------:|------:|
| Train | 8,383 | 8,383 |
| Val | 1,794 | 1,794 |
| Test | 1,802 | 1,802 |

## Classes

| ID | Class |
|:--:|:------|
| 0 | background |
| 1 | hazelnut |
| 2 | forest |
| 3 | permanent_cropland |
| 4 | greenhouse |
| 5 | grassland |
| 6 | sparsely_vegetated_areas |
| 7 | arable_land |
| 8 | discontinuous_urban_fabric |
| 9 | road_and_rail_networks |
| 10 | water_courses |
| 11 | water_bodies |
| 12 | wetland |

## Evaluated Models

- DeepLabV3+
- SegFormer
- UNet++
- UNetFormer

## Notes

Large raw data files and trained model checkpoints are intentionally not tracked in this Git repository. This repository contains documentation, scripts, metrics, confusion matrices, and selected visual outputs only.
