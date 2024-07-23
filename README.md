## Machine Unlearning Algorithms


### Unlearned Demo
|  Model | Dataset |  Link  |
| :----: | :-----: | :----: |
|**Computer Vision** |
| Vision Transformer | tiny ImageNet |  |
| 3-layer FNN | Mnist |  |


### Unlearning Algorithms

| Algorithm |  Cite  | Progress |
| :------:  | :----: | :------: |
| Selective Gradient Ascent | | done |


| Forgetten Data Type |  | Progress |
| :------:      | :----: | :------: |
| Class-Wise Unlearning | | done |
| Random Sample Unlearning | |done |

### Unlearning Evaluation Metrics

#### Unlearning Quality

| Metrics|    Cite   | Progress  | 
| :----: | :-------: | :-------: |
| Entropy of Confidence for Model Output |figure 4 [Eternal Sunshine of the Spotless Net: Selective Forgetting in Deep Networks](https://arxiv.org/abs/1911.04933) | done |
Accuracy on Forgetten Data |  | done |
MIA Scores |C.3 [Model Sparsity Can Simplify Machine Unlearning](https://arxiv.org/abs/2304.04934) | done | 
Re-learning Time   | | done    |
Model Distribution | | planned |

#### Unlearning Efficiency
How efficient to unlearn a model compared to retrain a model?

| Metrics|    Cite   | Progress  | 
| :----: | :-------: | :-------: |  
| Unlearning Time |  | done |
| Passed Samples |  | done |
#### Unlearned Model Utility
How accurate and generalize to retain data the unlearned model is?

| Metrics|    Cite   | Progress  | 
| :----: | :-------: | :-------: | 
Accuracy on Retain and Test Data |  | done |


### Reference:

Collection:

[tamlhp/awesome-machine-unlearning: Awesome Machine Unlearning (A Survey of Machine Unlearning)](https://github.com/tamlhp/awesome-machine-unlearning?tab=readme-ov-file)
[jjbrophy47/machine_unlearning: Existing Literature about Machine Unlearning](https://github.com/jjbrophy47/machine_unlearning)


[[ICLR24] SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation](https://www.optml-group.com/posts/salun_iclr24)


Recommender Systems:

[[2403.06737] Post-Training Attribute Unlearning in Recommender Systems](https://arxiv.org/abs/2403.06737) 