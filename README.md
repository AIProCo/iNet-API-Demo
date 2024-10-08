# iNet-API-Demo (Demo program for iNet-API)

- This is a repository for demo of iNet-API developped by AIPro Inc.
  + iNet: AIPro Deep Learning Solution supporting the following functionalities:
     - Object Detection 
     - Object Tracking
     - Pedestrian Attribute Recognition
     - Pedestrian Counting & Zone Hitmap Estimation
     - Fire Classification
     - Crowd Counting

# Licensing and Restrictions

- Commercial use must be approved by AIPro Inc. 
- The inference speed is limited to 10fps for each channel.
- The maximum number of inferences for all channels is set to 10000.
- (Important!!) there are several hidden features to prevent illegal use in this repository.
  
------------------

### **Dependency**

- cuda 12.1.0 (cuda_12.1.0_531.14_windows.exe)
- cuDNN 8.9.6 (cudnn-windows-x86_64-8.9.6.50_cuda12-archive.zip)
- openvino 2023.2.0

### **Installation and Solution Guide**

- You can refer to one of the following technical documents in the repository for installation and guide:
  + Korean: AIPro_iNet_solution_guide_v1.4.x(Korean).pdf

### **Download and extract files**
- Download and upzip the followings zip files. Then, copy and paste bin, inputs, and videos directories to the solution directory (the directory including the `.sln` file):
  + Cuda compute capability of your GPU should be 8.6 (ex: RTX 30xx): 
    - https://drive.google.com/file/d/1UVovFfOoeOz6X8Db195FUeb8nwAbWTsW/view?usp=sharing

### **Run the project**

- Open the following sln:
  + `iNet-API-Demo.sln` in the solution directory
