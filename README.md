# AV Sensor Fault Injection Toolkit 🚗⚡

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

## Data

The toolkit operates on the [Boreas dataset](https://www.boreas.utias.utoronto.ca).
Data are **not** included in this repository; download the sequences you need and
check the dataset license. Another dataset only requires a loader for its format.

---
## ⚖️ License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
