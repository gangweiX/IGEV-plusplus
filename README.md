# IGEV++: Iterative Multi-range Geometry Encoding Volumes for Stereo Matching
Gangwei Xu, Xianqi Wang, Zhaoxing Zhang, Junda Cheng, Chunyuan Liao, Xin Yang <br>

## Network architecture
![teaser](figures/network.png)
The IGEV++ first builds Multi-range Geometry Encoding Volumes (MGEV) via Adaptive Patch Matching (APM). MEGV encodes coarse-grained geometry information of the scene for textureless regions and large disparities and fine-grained geometry information for details and small disparities after 3D aggregation or regularization. Then we regress an initial disparity map from MGEV through $soft \; argmin$, which serves as the starting point for ConvGRUs. In each iteration, we index multi-range and multi-granularity geometry features from MGEV, selectively fuse them, and then input them into ConvGRUs to update the disparity field.

## Visual comparisons with SOTA methods in large disparities.
![image](figures/teaser.png)
PCWNet is a volume filtering-based method, DLNR is an iterative optimization-based method, and GMStereo is a transformer-based method. They all struggle to handle large disparities in large textureless objects at a close range.
