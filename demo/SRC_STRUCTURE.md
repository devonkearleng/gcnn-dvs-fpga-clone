# Demo Source Structure

This directory contains multiple dataset-specific implementations of the FPGA inference demo.

## Folders

### `src/` - Original Reference
The base implementation kept for reference. Configured for **10-class inference** (MNIST-DVS/CIFAR10-DVS).

### `src_mnistdvs/` - MNIST-DVS Demo
Customized for **MNIST-DVS** (10 classes: 0-9 digits)
- Weight matrix: `4096 × 10`
- Expected manifest: `sd_export/mnistdvs/all/manifest.txt`
- Event samples: `sd_export/mnistdvs/all/*.txt`
- Use with: `mw_mnistdvs_*.txt` (qat_r3, qat_r5, float_r3, float_r5)

### `src_cifar10/` - CIFAR10-DVS Demo
Customized for **CIFAR10-DVS** (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Weight matrix: `4096 × 10`
- Expected manifest: `sd_export/cifar10/train/manifest.txt`
- Event samples: `sd_export/cifar10/train/*.txt`
- Use with: `mw_cifar10_*.txt` (qat_r3, qat_r5, float_r3, float_r5)

### `src_ncars/` - N-Cars Demo
Customized for **N-Cars** (2 classes: car, background)
- Weight matrix: `4096 × 2` ← **Key difference: 2 classes instead of 10**
- Expected manifest: `sd_export/ncars/train/manifest.txt` or `sd_export/ncars/test/manifest.txt`
- Event samples: `sd_export/ncars/train/*.txt` or `sd_export/ncars/test/*.txt`
- Use with: `mw_ncars_*.txt` (qat_r3, qat_r5, float_r3, float_r5)
- Changes from base:
  - `weights_data[4096][2]` instead of `[4096][10]`
  - `output_dim = 2` instead of `10`
  - `output_vals[2]` instead of `[10]`
  - Argmax loop: `for(int i = 0; i < 2; i++)` instead of `for(int i = 0; i < 10; i++)`

## Compilation & Deployment

Each folder can be compiled independently for ZCU102:
1. Copy desired `src_*` folder contents to FPGA build system
2. Ensure corresponding `mw_*.txt` weights are on SD card at `0:/mw.txt`
3. Mount dataset with `manifest.txt` at `0:/manifest.txt`
4. Compile and deploy to ZCU102
5. Run and collect accuracy metrics

## Weight Files Reference

Located in `SW/weights/`:
- **MNIST-DVS**: `mnistdvs/mw_mnistdvs_{float,qat}_{r3,r5}.txt`
- **CIFAR10-DVS**: `cifar10/mw_cifar10_{float,qat}_{r3,r5}.txt`
- **N-Cars**: `ncars/mw_ncars_{float,qat}_{r3,r5}.txt`

## Event Export Paths

Located in `sd_export/`:
- **MNIST-DVS**: `mnistdvs/all/` (30K samples + manifest.txt + labels.txt)
- **CIFAR10-DVS**: `cifar10/train/` (10K samples + manifest.txt + labels.txt)
- **N-Cars**: `ncars/train/` (15K+ train) and `ncars/test/` (8K+ test)
