# Hospital3D: Synthetic PointCloud Perception for Safer Hospital Robot Navigation  

## 📌 Overview  
Hospital environments are full of dynamic, cluttered, and privacy-sensitive elements such as **beds, IV poles, wheelchairs, trolleys, and curtains**. Training robots to navigate safely in such spaces requires robust perception systems — but collecting real hospital data is difficult due to **privacy, cost, and ethical constraints**.  

This project tackles the challenge by building a **synthetic dataset pipeline** and training a **PointNeXt-tiny** model for **3D object detection in point clouds**, integrated into the **ROS 2 Nav2 navigation stack**.  

---

## 🚀 Motivation  
- Real hospital data is **hard to collect** due to privacy and safety restrictions.  
- Robots need to **detect and avoid hospital-specific objects** in real-time.  
- Synthetic data provides a **cheap, safe, and scalable alternative**.  

---

## 🏗️ Pipeline  

1. **Image Collection**  
   - 2D reference images of **38 hospital-relevant objects** were collected.  

2. **3D Object Generation**  
   - AI-based tools (Photogrammetry, Tripo-AI) converted 2D images → **3D meshes (.fbx/.obj)**.  

3. **Scene Simulation**  
   - **BlenderProc** used for randomized rendering (lighting, angles, object positions).  
   - Depth maps generated and converted into **3D point clouds (.ply → .npy)**.  

4. **Final Dataset**  
   - **1140 scans** across **38 object categories**.  
   - Standardized to **1024 points per scan**.  

---

## 🤖 AI Model: PointNeXt-tiny  

- A **lightweight 3D deep learning model** trained to classify hospital objects and regress 3D bounding boxes.  
- Input: **1024 sampled points per object**.  
- Output:  
  - **Class label** (e.g., hospital bed, IV pole).  
  - **3D bounding box** (center, size, orientation).  

### Training Setup  
- **Hardware:** AMD Ryzen CPU, 16GB RAM, Ubuntu.  
- **Framework:** PyTorch + Open3D.  
- **Epochs:** 30  
- **Batch size:** 32  
- **Augmentations:** jitter, dropout, random rotation.  
- **Validation accuracy:** **87%**  

---

## 📊 Results  

- **High overall accuracy (87%)** across 38 classes.  
- Strong detection for large hospital objects (beds, carts).  
- **Challenging cases:** thin/ambiguous objects (IV poles, stands).  
- **Comparison with baseline:**  
  - PointNet: **75%** accuracy.  
  - PointNeXt-tiny: **87%** accuracy.  

---

## 🦾 ROS 2 Integration  

The trained detector was integrated into the **ROS 2 Nav2 navigation framework**:  

1. **Depth camera** → point cloud stream.  
2. **Detector node** (ROS 2, rclpy) → runs PointNeXt-tiny inference.  
3. Publishes **/detected_obstacles** as polygons.  
4. **Nav2 costmap** updates with obstacles → robot avoids collisions.  

👉 This enables **hospital robots to detect and navigate around beds, carts, and equipment in real time** without privacy-sensitive data.  

---

## 🔑 Key Contributions  
✔️ Synthetic dataset pipeline for hospital robotics.  
✔️ PointNeXt-tiny model trained **CPU-only**, achieving 87% accuracy.  
✔️ ROS 2 integration showing practical use for **safe hospital navigation**.  
✔️ Framework adaptable to **warehouses, elder care, and service robots**.  

---

## 📚 References  
- Qi et al. *PointNet, PointNet++*. CVPR/NeurIPS.  
- Qian et al. *PointNeXt*. NeurIPS 2022.  
- Denninger et al. *BlenderProc*. ACCV 2019.  
- Macenski et al. *ROS2 Nav2*. ICRA 2020.  
- Matthews et al. *Synthetic Data for Healthcare AI*. Nature 2021.  

---

## 🙌 Acknowledgments  
- Supervisor: **Dr. Erivelton Nepomuceno** (Maynooth University).  
- GenAI tools (ChatGPT) for brainstorming, documentation, and code explanation support.  

--- 

