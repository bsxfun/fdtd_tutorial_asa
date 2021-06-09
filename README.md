Code for FDTD Tutorial, 180th ASA Meeting (Acoustics in Focus)
==============================================================

Matlab code for the Acoustics in Focus Invited Talk: 

*2pCA2. Tutorial on finite-difference time-domain (FDTD) methods for room acoustics simulation.*

##Usage

Run the script 'fdtd_tutorial_asa.m' in Matlab.

###Dependencies

* This code was tested in Matlab 2019b, but it code should run in various Matlab versions with CUDA/GPU acceleration disabled.

* For CUDA acceleration you will need a CUDA-supported Nvidia GPU with an up-to-date Nvidia driver, along with Matlab's Parallel Computing Toolbox.   

* CUDA support will be limited by your GPU architecture and Matlab version; see: [https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html](https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html).

* The supplied PTX file was compiled with CUDA toolkit version 10.2.

* PTX code can be recompiled with NVCC from the CUDA toolkit (install appropriate version for your GPU and Matlab version).
