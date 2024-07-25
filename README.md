## Machine Unlearning Algorithms


### Unlearned Demo
|  Model | Dataset |  Link  |
| :----: | :-----: | :----: |
|**Computer Vision** |
| Vision Transformer | tiny ImageNet, cifar100 |  |
| 3-layer FNN | Mnist |  |


### Unlearning Algorithms

| Algorithm |  Cite  | Progress |
| :------:  | :----: | :------: |
| Selective Gradient Ascent | | done |
| RMU | | planned |


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



### evaluate.py

This module provides a suite of functions for evaluating and comparing machine learning models for model unlearning algorithms.

Key Features:
- Accuracy calculation across different datasets
- Membership Inference Attack (MIA) scoring
- Entropy calculation and visualization
- Relearning time and computation
- Gradient Visualization

Main Functions:

- compare_mia_scores(model_list, model_names, forget_loader, retain_loader, unseen_loader)
    Compares Membership Inference Attack scores for multiple models.

- compare_accuracy(model_list, model_names, forget_loader, retain_loader, test_loader, unseen_loader)
    Compares accuracies of multiple models on different datasets.

- compare_entropy(model_list, model_names, forget_loader, retain_loader, unseen_loader)
    Compares prediction entropy of models across different datasets.

- compare_grad(modellist, modelnames, forget_loader, retain_loader, unseen_loader)
    Compares gradients of multiple models on different datasets.

- compare_relearn_time(model_list, model_names, forget_loader, test_loader, lr, momentum, optimizer_fn, acc_threshold=0.97)
    Evaluates how long it takes for models to relearn forgotten data.


### unlearn.py

This module implements selective gradient-based unlearning algorithms.
It aims to reduce model performance on specific data (forget set) while maintaining performance on retained data.

Key Features:
- Selective gradient computation
- Customizable gradient selection criteria
- Flexible parameter filtering for targeted unlearning

Main Functions:
- select_grads_fn(retain_grads, forget_grads, diff_threshold_ratio=0.8, magnitude_threshold_ratio=0.5)
    Selects gradients based on predefined criteria for unlearning.

- unlearn_selectiveGrad(model, retain_loader, forget_loader, test_loader, criterion, num_epochs, 
                        learning_rate=0.01, select_grads_fn=default_select_grads, 
                        filter_param_fn=None, ft_acc_threshold=0.1)
    Performs the unlearning process using selective gradient descent/ascent.

Usage:

    from unlearn import unlearn_selectiveGrad, select_grads_fn

    unlearned_model, results = unlearn_selectiveGrad(
        model, 
        retain_loader, 
        forget_loader, 
        test_loader,
        criterion, 
        num_epochs, 
        learning_rate,
        select_grads_fn=select_grads_fn
    )

