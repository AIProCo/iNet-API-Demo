# iNet-API-Demo (Demo program for iNet-API)

- This is a repository for demo of iNet-API developped by AIPro Inc.
  + iNet: AIPro Deep Learning Solution supporting the following functionalities:
     - Object Detection
     - Object Tracking
     - Pedestrian Attribute Recognition
     - Pose Estimation
     - Action Recognition
     - Pedestrian Counting & Zone Hitmap Estimation

- In this demo, the number of inferences is limited to 6000.
- Commercial use must be approved by AIPro Inc.
  
------------------

### **Dependency**

- Cuda 11.6.2
- cuDNN 8.4.0 (cudnn-windows-x86_64-8.4.0.27_cuda11.6)
- zlibwapi.dll (cuDNN 8.4.0 uses this)

### **Installation and Solution Guide**

- You can refer to one of the following technical documents in the repository for installation and guide:
  + English: AIPro_iNet_solution_guide_v1(English).pdf
  + Korean: AIPro_iNet_solution_guide_v1(Korean).pdf

### **Download and extract files**
- Download and upzip the followings zip files. Then, copy and paste bin, inputs, and videos directories to the solution directory (the directory including the `.sln` file):
  + Cuda compute capability of your GPU should be 8.6 or later(ex: RTX 30xx): 
    - https://drive.google.com/file/d/1NXW9zycWTg99Pw3iljoonEAAw_BFz-mU/view?usp=share_link

### **Run the project**

- Open the following sln:
  + `iNet-API-Demo.sln` in the solution directory
