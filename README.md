# IGEV++: Iterative Multi-range Geometry Encoding Volumes for Stereo Matching
Gangwei Xu, Xianqi Wang, Zhaoxing Zhang, Junda Cheng, Chunyuan Liao, Xin Yang <br>

## Network architecture
![teaser](figures/network.png)
The IGEV++ first builds Multi-range Geometry Encoding Volumes (MGEV) via Adaptive Patch Matching (APM). MEGV encodes coarse-grained geometry information of the scene for textureless regions and large disparities and fine-grained geometry information for details and small disparities after 3D aggregation or regularization. Then we regress an initial disparity map from MGEV through $soft \; argmin$, which serves as the starting point for ConvGRUs. In each iteration, we index multi-range and multi-granularity geometry features from MGEV, selectively fuse them, and then input them into ConvGRUs to update the disparity field.

## Comparisons with SOTA methods
![image](figures/teaser_v2.png)
\textbf{Left:} Comparisons with state-of-the-art stereo methods across different disparity ranges on the Scene Flow test set. Our IGEV++ outperforms previously published methods by a large margin across all disparity ranges. 
Right: Comparisons with state-of-the-art stereo methods on Middlebury and KITTI leaderboards.


## Visual comparisons with SOTA methods in large disparities.
![image](figures/teaser.png)
PCWNet is a volume filtering-based method, DLNR is an iterative optimization-based method, and GMStereo is a transformer-based method. They all struggle to handle large disparities in large textureless objects at a close range.

## Demos
Pretrained models can be downloaded from [google drive](https://drive.google.com/drive/folders/1SsMHRyN7808jDViMN1sKz1Nx-71JxUuz?usp=share_link)

We assume the downloaded pretrained weights are located under the pretrained_models directory.

You can demo a trained model on pairs of images. To predict stereo for demo-imgs directory, run
```Shell
python demo_imgs.py \
--restore_ckpt ./pretrained_models/igev_plusplus/sceneflow.pth \
```
