# GaitGuard
## 1. Overview
With the increasing prevalence of online videos featuring pedestrians, the gait information embedded in such content poses significant privacy risks. Previous gait anonymization methods suffer from poor visual naturalness in the synthesized gait and lack precise appearance control. To address these challenges, we present GaitGuard, the first diffusion model-based framework for gait anonymization and protection. Operating as a purely black-box attack method, GaitGuard requires no prior knowledge of the target model, distinguishing it from traditional adversarial approaches. GaitGuard incorporates an Appearance Fusion Module (AFM) and a Motion Fusion Module (MFM) to encode and blend appearance and motion features from different individuals in the latent space, thereby achieving precise control over the generated gait during the denoising process. This design effectively protects gait privacy from recognition systems while ensuring visual naturalness and frame-to-frame consistency in the generated gait video. Extensive experiments on the CASIA-B and OUMVLP datasets demonstrate substantial reductions in recognition accuracy across representative gait recognition models, confirming the robustness and effectiveness of GaitGuard.

## 2. Method
### 2.1 Introduction
![image](https://github.com/phtphtpht/GaitGuard/blob/main/image/1.png)

### 2.2 Workflow
![image](https://github.com/phtphtpht/GaitGuard/blob/main/image/2.png)

### 2.3 Architecture
![image](https://github.com/phtphtpht/GaitGuard/blob/main/image/3.png)

### 2.4 Part of Details
![image](https://github.com/phtphtpht/GaitGuard/blob/main/image/4.png)

### 2.5 Part of Experiments
![image](https://github.com/phtphtpht/GaitGuard/blob/main/image/5.png)
![image](https://github.com/phtphtpht/GaitGuard/blob/main/image/6.png)
![image](https://github.com/phtphtpht/GaitGuard/blob/main/image/7.png)

## 3. Features
The GaitGuard framework consists of four main stages:  
(1)Silhouette Extraction and Masking:  
Given frame $F_i$, we first apply an image segmentation model $M_S$ to obtain the gait silhouette $S_A = M_S(F_i)$. The corresponding masked frame is then computed as:  
$F_{mask} = F_i \cdot (1- S_A)$  

(2)Pose Estimation and Background Inpainting:  
We extract the pose sequence using model $M_E$ as:  
$P_i = M_E(F_i)$  
Concurrently, we reconstruct the background using an inpainting model $M_I$ as:  
$F_{bkg} = M_I(F_i,F_{mask})$  

(3)Gait Generation:  
The proposed GaitGen network $M_G$ synthesizes a new gait appearance $G$ by conditioning on two pose sequences (identity pose sequence $P_I$ and reference pose sequence $P_R$) along with two full-body images (identity image $I_R$ and reference image $I_I$). This process can be formally expressed as:  
$G = M_G(P_I,P_R, I_I I_R)$  

(4)Frame Composition:  
The generated gait $G$ is composited with the background $F_{bkg}$ to produce the anonymized frame $F_i'$  

(5)
The complete GaitGuard framework can be formalized as:  
$F_i' = GaitGuard(F_i, P_I, P_R, I_I, I_R)$  
where $F_i'$ represents the anonymized video frame.  

## 4. How to use?

## 5. Reference
[MooreThreads AnimateAnyone][https://github.com/MooreThreads/Moore-AnimateAnyone]  
[Lama][https://github.com/advimman/lama]  

