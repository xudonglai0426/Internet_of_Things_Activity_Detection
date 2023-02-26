# IoT-Activity-Detection
 The project is aimed at developing new tools for classifying videos of human-machine interactions in the Internet-of-Things (IOT) domain. Namely, given videos of humans interacting with IoT devices (i.e. smart appliances such as fridge, toaster, washing machines, Alexa, etc), the aim is to (1) design predictive video features, which (2) can be extracted efficiently in real-time to classify videos in terms of the activity being performed (opening or closing a fridge, loading or unloading a washing machine, etc.). The grand goal and motivation for the work is to generate labels for IoT network traffic, simply by training cameras onto IoT devices in the various IoT labs across US universities. Thus, the project aims to solve a main bottleneck in research at the intersection of Machine Learning and IoT, namely, the scarcity of labeled IoT traffic data to solve ML problems such as activity and anomaly detection using supervised or unsupervised detection procedures. 


> **Introduction borrowed from [Shangjie's Repository](https://github.com/aJay0422/IoT-Activity-Detection-Clean)**


## Dataset Introduction
Currently, we are using a dataset consists of 951 videos.  
First, we use [Detectron2](https://github.com/facebookresearch/detectron2) to extract human body keypoints feature.
In each frame, there are 17 keypoints represented by their 2D coordinates (x,y). For each video,
the number of feature is 17 * 2 * n_frames.  
Second, we interpolate the feature sequence to length 100, then we have 17 * 2 * 100 = 3400 features for each video.  
The interpolated data can be found in the folder [`feature_archive/all_feature_interp951.npz`]() or [download](https://drive.google.com/drive/folders/1Wmhi-ftV_buR9jFlPW0u4IR6F3idyr5G?usp=sharing)
here. 
```python
# How to use the dataset
import numpy as np
data_path = "feature_archive/all_feature_interp951.npz"
all_data = np.load(data_path, allow_pickle=True)
X_all = all_data["X"]   # shape(951, 3400)
Y_all = all_data["Y"]
X_all = X_all.reshape(-1, 100, 34).transpose(0, 2, 1)   # reshape into (951, 34, 100)
```
If you want to use the data in a "dataloader way", you can either construct your own dataset and dataloader or 
use the function `prepare_data` in the [Transformer/utils_weighted.py](https://github.com/kl3259/IoT-Activity-Detection/blob/main/doc/Transformer/utils_weighted.py), which returns a train dataloader, a validation dataloader and a test dataloader. We also provide the interface to generate dataloader with confidence score from Detectron2 in function `prepare_data_w_weight`.

## Transformer/model.py
This file is for building the Transformer model. Three functions `transformer_base`,
`transformer_large` and `transformer_huge` return three Transformer models with
different size(different number of parameters).  
```python
from Transformer.model import transformer_base, transformer_large, transformer_huge
model_base = transformer_base()
model_large = transformer_large()
model_huge = transformer_huge()
```

## Transformer/train.py
This file is for training a Transformer model. The model weights can be [downloaded](https://drive.google.com/drive/folders/1Wmhi-ftV_buR9jFlPW0u4IR6F3idyr5G?usp=sharing) here.
```python
import torch
from Transformer.model import transformer_base
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = transformer_base()
model.to(device)
weight_path = "path/to/weight"
model.load_state_dict(torch.load(weight_path, map_location=device))
```

## Confidence score
The idea is that our keypoint feature estimation may not be accurate. We can use the confidence scores detectron2 returned to measure how confident are we going to trust the feature, and calculate a single confidence score for each video. For those videos with high confidence, we should give them more weight when training.  

The confidence score provided in [`feature_archive/confidence_scores_by_frame.npy`](https://github.com/kl3259/IoT-Activity-Detection/blob/main/feature_archive/confidence_scores_by_frame.npy) is in the shape of __17 * n_frames__. Each confidence score demonstrates the confidence support for each keypoint in each frame of that video. In preprocessing phase, both the keypoints and the confidence score were interpolated along the temporal dimension. We finalize the input feature from arbitrary time length to 100 time steps. Therefore, the shape of the score of each video is __17 * 100__. Our goal is to turn these 1700 scores into one single score for each video, in a clever way.
  
The interpolated confidence scores can also be [downloaded](https://drive.google.com/drive/folders/1Wmhi-ftV_buR9jFlPW0u4IR6F3idyr5G?usp=sharing) here.

```python
# How to use interpolated confidence score
import numpy as np
scores = np.load("feature_archive/confidence_scores_by_frame.npy")   # shape(951, 100, 17)
```


## Transformer/train_weighted.py
To use the weights in the training process, we provide a new dataloader and a new initialization method with weight in the function [`prepare_data_w_weight`](https://github.com/kl3259/IoT-Activity-Detection/blob/main/doc/Transformer/utils_weighted.py). It can generate `X_batch`, `Y_batch`, and `weight_batch`. Please see the details in `mydataset_w_weight` in [Transformer/train_weighted.py](https://github.com/kl3259/IoT-Activity-Detection/blob/main/doc/Transformer/train_weighted.py) for the newly-added weight.

We try to train the transformer in an iterative fashion. First, we train the transformer with unweighted data (initial weights set to be 1 for all samples) and then get the attention. For the second phase of training, we use the attention/margin/entropy generated from the previous model to compute the confidence score as the weight for each training video, and we train the model again by using weighted loss (weighted cross-entropy), the model also generates new attention/margin/entropy for each sample. Next, we repeat the process several times to see whether this training procedure can increase the test accuracy on all of the test videos or not. In this section, we compare different definitions of weight. The function `train_w_weight` in [`Transformer/train_weighted.py`]()  is the main training interface. The function `train_transformer_w_weight` run one iteration for weighted training. The function `train_transformer_iter` automatically runs iterations with updating weights at the end of each iteration. 

### **Weights**

1. **Attention(Attention-based confidence score)**
    ```python
    pseudo code
    Input: 
    1. confidence_by_keypoint # a tensor of size (n_samples, n_frames, n_keypoints) -> average through keypoints -> (n_samples, n_frames)
    2. attention_by_frame # (n_samples, n_heads, n_frames, n_frames) -> average through attention heads -> (n_samples, n_frames, n_frames) -> average through frames -> (n_samples, n_frames)
    Algorithm:
    confidence_score = confidence_by_frame * attention_by_frame # elementwise product
    confidence_score = np.mean(confidence_score, axis = (1))
    Output: confidence_score # a tensor of size (n_samples,)
    ```
   In our case, we have n_frames = 100, n_kepoints = 17, n_samples = 951

2. **Margin(Classification Margin)**
    Suppose $f$ is the trained model, which outputs a probability distribution $p_1, p_2, ..., p_K$ for each input sample. Then the margin is defined as:
    $$Margin = max(p_1, p_2, ..., p_K) - 2^{nd}max(p_1, p_2, ..., p_K)$$

3. **Entropy(Classification Entropy)**
    For the probability distribution $p_1, p_2, ..., p_K$ for each input sample, we can also measure the confidence by entropy. The lower the entropy is, the more confident a prediction would be. The definition is:
    $$Entropy = -\sum_{i = 1}^{K}p_i\log_2{p_i}$$



