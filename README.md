---
library_name: transformers
license: other
base_model: nvidia/mit-b2
tags:
- image-segmentation
- vision
- generated_from_trainer
model-index:
- name: mit-b2-crop_v0
  results: []
---

# mit-b2-crop_v0

This model is a fine-tuned version of [nvidia/mit-b2](https://huggingface.co/nvidia/mit-b2) on the Sowmith1999/agaid_residue_only dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2808
- Mean Iou: 0.3806
- Mean Accuracy: 0.7611
- Overall Accuracy: 0.7611
- Accuracy Background: nan
- Accuracy Residue: 0.7611
- Iou Background: 0.0
- Iou Residue: 0.7611

## Training and evaluation data

Residue Only data

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 6e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 1337
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: polynomial
- training_steps: 1000
Here Mean IOU is just the half of the IOU Residue as IOU background is zero.
![IOU plots](./mit-b2-crop_v0.png)

### Training results

| Training Loss | Epoch   | Step | Validation Loss | Mean Iou | Mean Accuracy | Overall Accuracy | Accuracy Background | Accuracy Residue | Iou Background | Iou Residue |
|:-------------:|:-------:|:----:|:---------------:|:--------:|:-------------:|:----------------:|:-------------------:|:----------------:|:--------------:|:-----------:|
| No log        | 1.0     | 37   | 0.4510          | 0.1649   | 0.3297        | 0.3297           | nan                 | 0.3297           | 0.0            | 0.3297      |
| No log        | 2.0     | 74   | 0.4209          | 0.1916   | 0.3832        | 0.3832           | nan                 | 0.3832           | 0.0            | 0.3832      |
| No log        | 3.0     | 111  | 0.3446          | 0.3684   | 0.7368        | 0.7368           | nan                 | 0.7368           | 0.0            | 0.7368      |
| No log        | 4.0     | 148  | 0.3653          | 0.3379   | 0.6759        | 0.6759           | nan                 | 0.6759           | 0.0            | 0.6759      |
| No log        | 5.0     | 185  | 0.3450          | 0.2673   | 0.5346        | 0.5346           | nan                 | 0.5346           | 0.0            | 0.5346      |
| No log        | 6.0     | 222  | 0.3373          | 0.3349   | 0.6698        | 0.6698           | nan                 | 0.6698           | 0.0            | 0.6698      |
| No log        | 7.0     | 259  | 0.3591          | 0.2551   | 0.5103        | 0.5103           | nan                 | 0.5103           | 0.0            | 0.5103      |
| No log        | 8.0     | 296  | 0.3767          | 0.3090   | 0.6180        | 0.6180           | nan                 | 0.6180           | 0.0            | 0.6180      |
| No log        | 9.0     | 333  | 0.3993          | 0.3893   | 0.7787        | 0.7787           | nan                 | 0.7787           | 0.0            | 0.7787      |
| No log        | 10.0    | 370  | 0.3244          | 0.3528   | 0.7057        | 0.7057           | nan                 | 0.7057           | 0.0            | 0.7057      |
| 0.3601        | 11.0    | 407  | 0.3343          | 0.3610   | 0.7219        | 0.7219           | nan                 | 0.7219           | 0.0            | 0.7219      |
| 0.3601        | 12.0    | 444  | 0.3271          | 0.3243   | 0.6486        | 0.6486           | nan                 | 0.6486           | 0.0            | 0.6486      |
| 0.3601        | 13.0    | 481  | 0.2944          | 0.3653   | 0.7307        | 0.7307           | nan                 | 0.7307           | 0.0            | 0.7307      |
| 0.3601        | 14.0    | 518  | 0.3926          | 0.2846   | 0.5693        | 0.5693           | nan                 | 0.5693           | 0.0            | 0.5693      |
| 0.3601        | 15.0    | 555  | 0.2980          | 0.4201   | 0.8403        | 0.8403           | nan                 | 0.8403           | 0.0            | 0.8403      |
| 0.3601        | 16.0    | 592  | 0.3199          | 0.3514   | 0.7029        | 0.7029           | nan                 | 0.7029           | 0.0            | 0.7029      |
| 0.3601        | 17.0    | 629  | 0.3191          | 0.3884   | 0.7768        | 0.7768           | nan                 | 0.7768           | 0.0            | 0.7768      |
| 0.3601        | 18.0    | 666  | 0.3116          | 0.3859   | 0.7718        | 0.7718           | nan                 | 0.7718           | 0.0            | 0.7718      |
| 0.3601        | 19.0    | 703  | 0.2945          | 0.4122   | 0.8245        | 0.8245           | nan                 | 0.8245           | 0.0            | 0.8245      |
| 0.3601        | 20.0    | 740  | 0.3131          | 0.3856   | 0.7711        | 0.7711           | nan                 | 0.7711           | 0.0            | 0.7711      |
| 0.3601        | 21.0    | 777  | 0.3001          | 0.3702   | 0.7404        | 0.7404           | nan                 | 0.7404           | 0.0            | 0.7404      |
| 0.2434        | 22.0    | 814  | 0.2842          | 0.3605   | 0.7210        | 0.7210           | nan                 | 0.7210           | 0.0            | 0.7210      |
| 0.2434        | 23.0    | 851  | 0.2849          | 0.3626   | 0.7252        | 0.7252           | nan                 | 0.7252           | 0.0            | 0.7252      |
| 0.2434        | 24.0    | 888  | 0.2799          | 0.3868   | 0.7735        | 0.7735           | nan                 | 0.7735           | 0.0            | 0.7735      |
| 0.2434        | 25.0    | 925  | 0.2902          | 0.3635   | 0.7270        | 0.7270           | nan                 | 0.7270           | 0.0            | 0.7270      |
| 0.2434        | 26.0    | 962  | 0.2723          | 0.3859   | 0.7719        | 0.7719           | nan                 | 0.7719           | 0.0            | 0.7719      |
| 0.2434        | 27.0    | 999  | 0.2801          | 0.3855   | 0.7711        | 0.7711           | nan                 | 0.7711           | 0.0            | 0.7711      |
| 0.2434        | 27.0270 | 1000 | 0.2808          | 0.3806   | 0.7611        | 0.7611           | nan                 | 0.7611           | 0.0            | 0.7611      |

#### Confusion Matrix
|  | Positive | Negative |
| --------------- | --------------- | --------------- |
| Positive | 67.4 | 5.3 |
| Negative | 5.3 | 22 |


### Framework versions

- Transformers 4.49.0.dev0
- Pytorch 2.5.1+cu124
- Datasets 3.2.0
- Tokenizers 0.21.0
