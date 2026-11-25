# AV Sensor Fault Injection Toolkit 🚗⚡

[![arXiv](https://img.shields.io/badge/arXiv-2611.31667-b31b1b.svg)](https://arxiv.org/abs/2611.31667)

An open-source fault injection framework for autonomous vehicle (AV) perception systems. This toolkit operationalizes a cross-modal taxonomy of exteroceptive sensor faults, allowing researchers to systematically inject phenomenological, symptom-level degradations into time-stamped **Camera**, **LiDAR**, and **Radar** datasets. 

It is designed to generate controlled testbeds and fault-annotated datasets for evaluating robust multi-sensor fusion and developing Fault Detection and Diagnosis (FDD) architectures.

---

## ✨ Features

- 📸 **Cross-Modal Support:** Operates natively on RGB images, 3D point clouds (`.bin`), and 2D polar radar power spectra (`.png` Navtech/Boreas format).
- 🧩 **Origin-Driven Taxonomy:** Includes **94 injection routines** mapped across 5 root-cause categories:
  - **Hardware** (e.g., lens cracks, LNA degradation, photodetector noise)
  - **Environmental** (e.g., rain, fog, multipath ghosts, window contamination)
  - **Calibration** (e.g., boresight offsets, thermal drift, extrinsic misalignment)
  - **Software** (e.g., packet loss, silent failures, inference timing delays)
  - **Synergistic** (e.g., thermal deformation + multipath)
- 🎛️ **Graded Severity & Burst Scheduling:** Four severity levels (*mild*, *moderate*, *severe*, *extreme*). Faults are scheduled in contiguous temporal bursts, supporting progressive degradation.
- 🏷️ **Ground-Truth Manifests:** Automatically generates a detailed `labels.csv` per run, recording the exact fault category, affected subsystem, severity, and deterministic seeds for every frame.

---

## 🛠️ Installation & Usage

Clone the repository and install the dependencies:

```bash
git clone https://github.com/davidepossenti/AV-SensorFault-Toolkit.git
cd AV-SensorFault-Toolkit
pip install -r requirements.txt
```
---

## 🚀 Usage Examples

The toolkit provides modular injectors for each sensor, executable directly from the command line.

### 1. Injecting Faults

**Camera:**
```bash
python -m camera.camera_injector \
    --in-dir /path/to/boreas/camera \
    --out-dir output/camera \
    --fault environment:rain \
    --fault hardware:dead_pixels \
    --balance-ratio 0.5 \
    --n-bursts 2 \
    --burst-len-min 10 \
    --burst-len-max 20 \
    --seed 42
```

**LiDAR:**
```bash
python -m lidar.lidar_injector \
    --in-dir /path/to/boreas/lidar \
    --out-dir output/lidar \
    --fault hardware:laser \
    --fault synergistic:scanning_fog \
    --balance-ratio 0.5 \
    --seed 42
```

**Radar:**
```bash
python -m radar.radar_injector \
    --in-dir /path/to/boreas/radar \
    --out-dir output/radar \
    --fault environment:multipath \
    --fault hardware:lna \
    --balance-ratio 0.5 \
    --seed 42
```

*Note: Temporal injection dynamics can be finely tuned using flags such as `--first-n`, `--n-bursts`, `--burst-len-min`, `--burst-len-max`, `--keep-head`, and `--keep-tail`.*

### 2. Visualization Tools

To verify the injected degradations side-by-side with the original clean data, you can use the included visualization tools:

```bash
# LiDAR side-by-side visualizer
python -m tools.visualize_lidar_faults \
    --clean-dir /path/to/boreas/lidar \
    --corrupted-dir output/lidar \
    --max-figures 5
```
--- 

## Data

The toolkit operates on the [Boreas dataset](https://www.boreas.utias.utoronto.ca).
Data are **not** included in this repository; download the sequences you need and
check the dataset license. Another dataset only requires a loader for its format.


## 📝 Citation

If you find this taxonomy or toolkit useful in your research, please consider citing our paper:

```bibtex
@misc{possenti2026exteroceptivesensorfaultsautonomous,
      title={Exteroceptive Sensor Faults in Autonomous Vehicles: A Cross-Modal Taxonomy and Fault Injection Toolkit}, 
      author={Davide Possenti and Daniele Cianca and Stefano Arrigoni and Francesco Braghin},
      year={2026},
      eprint={2611.31667},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={[https://arxiv.org/abs/2611.31667](https://arxiv.org/abs/2611.31667)}, 
}
```

---

---
## ⚖️ License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
