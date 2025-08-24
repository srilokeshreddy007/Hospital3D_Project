# Hospital3D_Project

# 🏥 Hospital 3D: Synthetic Point Cloud Perception for Safe Hospital Robot Navigation

**Author:** Sri Lokesh Reddy Anikreddypalli  
**Supervisor:** Dr. Erivelton Nepomuceno  
**Affiliation:** Maynooth University  

---

## 📌 Motivation

Hospitals are **dynamic and constrained environments** where mobile robots must navigate around both **static** (beds, carts, walls) and **dynamic soft objects** (curtains, IV poles, people).  
- Collecting **real 3D datasets** in hospitals is nearly impossible due to **privacy, safety, and logistical constraints**.  
- Synthetic data offers a **cheap, safe, and scalable** alternative.  

This project develops a **synthetic dataset pipeline** and a **PointNeXt-tiny 3D detection model** to enable **safe navigation in hospitals**. The system is designed to integrate into **ROS 2 Nav2**, allowing robots to perceive obstacles and plan safe paths.

---

## 📂 Project Structure

hospital3d_project/
│
├── backbones/ # PointNeXt-tiny backbone
├── configs/ # YAML configs (default + sota)
├── dataset_pipeline/ # Scripts for data generation (AI → 3D → BlenderProc)
├── figs/ # Figures for report/poster
├── nav2_config/ # Example Nav2 configs
├── outputs/ # Trained models & logs (excluded in GitHub)
├── ros2_nodes/ # ROS 2 detector node wrapper
├── runs/ # Training runs (excluded in GitHub)
├── src/ # Core training, inference, visualization
│ ├── train_sota.py
│ ├── eval.py
│ ├── infer_sota.py
│ ├── viz_offscreen.py
│ ├── model_sota.py
│ ├── dataset.py
│ └── utils.py
│
├── requirements.txt
├── README.md
├── labels.txt # Class labels (38 hospital objects)
├── train.txt # Train split
├── val.txt # Validation split
└── boxes.json # Ground truth box annotations



---

## 🛠️ Installation

Tested on **Ubuntu 22.04, Python 3.9, CPU-only**.

```bash
# clone
git clone https://github.com/<your-username>/hospital3d_project.git
cd hospital3d_project

# create virtual environment
python3 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
---


