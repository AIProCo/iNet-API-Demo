# iNet-API-Demo (Demo program for iNet-API)

- This is a repository for demo of iNet-API developped by AIPro Inc.
  + iNet: AIPro Deep Learning Solution supporting the following functionalities:
     - Object Detection and Tracking (RGB and Thermal)
     - Pedestrian Attribute Recognition
     - Pedestrian Counting & Zone Hitmap Estimation
     - Fire Classification
     - Crowd Counting

# Licensing and Restrictions

- Commercial use must be approved by AIPro Inc. 
- The maximum number of inferences for all channels is limited.
- (Important!!) there are several hidden features to prevent illegal use in this repository.
  
------------------

### **Dependency**

- cuda 12.9.0 (cuda_12.9.0_576.02_windows.exe)
- cuDNN 9.12.0 (cudnn_9.12.0_windows.exe)
- openvino 2023.2.0

### **Installation and Solution Guide**

- You can refer to one of the following technical documents in the repository for installation and guide:
  + Korean: AIPro_iNet_solution_guide_v1.4.8(Korean).pdf

### **Download and extract files**
- Download and upzip one of the followings zip files. Then, copy and paste bin, inputs, and videos directories to the solution directory (the directory including the `.sln` file):
  + Cuda compute capability of your GPU should be 8.6(RTX-30xx) or 8.9(RTX-40xx): 
    - RTX-30xx: https://drive.google.com/file/d/1x3LWVc07iyINwRn1HAACaO2eKFI4fcIY/view?usp=sharing
    - RTX-40xx: https://drive.google.com/file/d/1bY1ltfKbiYvEknobUdOYS7Zq2flZlrsT/view?usp=sharing
    
### **Run the project**

- Open the following sln:
  + `iNet-API-Demo.sln` in the solution directory
