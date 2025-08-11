# BIM_Updating
This repository contains a complete implementation method proposed in the paper for semi-automated local updating for as-built BIM of piping systems using point cloud data.
# Semi-Automated As-Built BIM Updating for Piping Systems

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/RainGo111/BIM_updating.svg)](https://github.com/RainGo111/BIM_updating/issues)
[![GitHub Stars](https://img.shields.io/github/stars/RainGo111/BIM_updating.svg)](https://github.com/RainGo111/BIM_updating/stargazers)

This repository contains the complete implementation of methods proposed in our paper for **semi-automated local updating of as-built BIM of piping systems using point cloud data**.

## 🎯 Overview

Our approach addresses the challenge of maintaining accurate as-built BIM models by automatically detecting and quantifying geometric changes in piping systems through point cloud analysis.

### Key Features
- **Automated piping segmentation**: Direct extraction of piping networks from point clouds
- **Geometric change detection**: Quantification of changes in length, height, radius, and angle
- **As-designed vs As-built comparison**: Systematic analysis of design deviations

## 🏗️ Methodology

### 1. Point Cloud Preprocessing
- **MATLAB Version**: 
  - Data processing for as-built point cloud models
  - Planar object segmentation
  - Multi-elevation piping system segmentation
- **PointNet++ Version**: 
  - Custom-trained network for direct piping segmentation
  - Real-time piping network extraction

### 2. Spatial and Topological Analysis
- **Geometric Parameter Extraction**: Automated measurement of pipe dimensions
- **Change Quantification**: Statistical analysis of geometric deviations

### 3. As-built BIM generation
- **As-designed Processing**: Revit model parameter extraction by using Dynamo
- **As-built Processing**: Point cloud-based geometric analysis

## 🚀 Quick Start

### Prerequisites
```bash
# Python environment
conda create -n bim-updating python=3.8
conda activate bim-updating

# MATLAB (for spatial analysis module)
MATLAB R2023b or later with Computer Vision Toolbox
```

## 📊 Results

Our method achieves:
- **Segmentation Accuracy**: 96% for piping system segmantation
- **Geometric Precision**: ±5mm for dimensional measurements
- **Processing Speed**: 70% faster than manual methods

## 🔧 Technical Details

### Algorithms
- **PointNet++**: Modified architecture for piping-specific features
- **RANSAC/ICP**: Robust geometric fitting and alignment
- **Connected Components**: Instance segmentation of individual pipes

### Data Formats
- **Input**: PLY, PCD, LAS point clouds
- **Output**: JSON geometric parameters, CSV change reports
- **Visualization**: 3D interactive models

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Real Point Cloud Data: Self-collected by the authors using laser scanning equipment
Synthetic Point Cloud Data: Generated using [Blensor](https://www.blensor.org/) simulation framework
Real BIM Models: Self-constructed Revit models by the authors
Simulation BIM Models: Sourced from [SimAUD Dataset](https://www.simaud.org/datasets/)

