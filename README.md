# SVD-GCN
Codes for CIKM 2022 paper SVD-GCN: A Simplified Graph Convolution Paradigm for Recommendation

# Environment
The algorithm is implemented in Python 3.8.5, with the following libraries additionally needed to be installed:<br/>
* Pytorch+GPU==1.8.0<br/>
* Numpy==1.19.2<br/>
* Pandas==1.1.4<br/>

# RUn the Algorithm

1. Run preprocess.py to get the required number of singular vectors/values. To make the calculated singular value/vectors more accurate,
q is expected to set (slightly) larger than req_vec (K in the paper).
2. Run SVD-GCN variants. SVD-GCN-B/U/I/M needs training, while SVD-GCN-S does not require any optimiziation. Param_Settings includes parameter settings for datasets used in this work. 
