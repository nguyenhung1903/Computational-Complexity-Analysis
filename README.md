# Computational Complexity Analysis

This repository provides tools to analyze and benchmark the computational complexity and GPU memory usage of various deep learning models, specifically for image restoration tasks. The main focus is on comparing different models in terms of FLOPs, parameter count, and VRAM usage.

## Contents
- `complexity_flops_analysis.py`: Analyzes the FLOPs (floating point operations) and parameter count of several models using [ptflops](https://github.com/sovrasov/flops-counter.pytorch).
- `benchmark_vram_usage.py`: Benchmarks the GPU memory (VRAM) usage of the same models during inference.

## Supported Models
The scripts expect the following models to be available in a `models` directory:
- Restormer
- PromptIR
- HAIR
- AdaIR
- QuaHaarIR

> **Note:** The actual model implementations are not included in this repository. You must provide them in the `models` directory with the correct class names as referenced in the scripts.

## Requirements
- Python 3.7+
- [PyTorch](https://pytorch.org/) (with CUDA support for GPU benchmarking)
- [ptflops](https://github.com/sovrasov/flops-counter.pytorch)

Install dependencies with:
```bash
pip install torch ptflops
```

## Usage

### 1. Model Complexity Analysis
Run the following command to print the FLOPs and parameter count for each model:
```bash
python complexity_flops_analysis.py
```

### 2. VRAM Usage Benchmark
Run the following command to print the GPU memory usage for each model:
```bash
python benchmark_vram_usage.py
```

## Customization
- To add or remove models, edit the `models` list in each script.
- To change the input size, modify the `input_shape` in `complexity_flops_analysis.py` and the tensor shape in `benchmark_vram_usage.py`.

## Notes
- Ensure your CUDA device is available and properly configured for GPU benchmarking.
- The scripts assume each model can be instantiated without arguments (except for `decoder=True` where specified).

## License
This project is for research and educational purposes.

## Citation
If you use this code in your research, please cite:
```
@misc{computational_complexity_analysis,
  author = {An Hung Nguyen},
  title = {Computational Complexity Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/nguyenhung1903/Computational_Complexity_Analysis}
}
```
