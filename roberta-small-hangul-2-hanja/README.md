---
library_name: transformers
base_model: klue/roberta-small
tags:
- generated_from_trainer
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: roberta-small-hangul-2-hanja
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-small-hangul-2-hanja

This model is a fine-tuned version of [klue/roberta-small](https://huggingface.co/klue/roberta-small) on the None dataset.
It achieves the following results on the evaluation set:
- Accuracy: 0.9918
- F1: 0.9824
- Loss: 0.0868
- Precision: 0.9814
- Recall: 0.9835

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 64
- eval_batch_size: 64
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 20

### Training results

| Training Loss | Epoch | Step | Accuracy | F1     | Validation Loss | Precision | Recall |
|:-------------:|:-----:|:----:|:--------:|:------:|:---------------:|:---------:|:------:|
| No log        | 1.0   | 482  | 0.8993   | 0.4607 | 0.8157          | 0.6392    | 0.3602 |
| 1.7229        | 2.0   | 964  | 0.9606   | 0.8537 | 0.4586          | 0.8728    | 0.8354 |
| 0.6279        | 3.0   | 1446 | 0.9701   | 0.9159 | 0.3291          | 0.9244    | 0.9075 |
| 0.418         | 4.0   | 1928 | 0.9761   | 0.9426 | 0.2596          | 0.9439    | 0.9413 |
| 0.316         | 5.0   | 2410 | 0.9797   | 0.9530 | 0.2159          | 0.9545    | 0.9515 |
| 0.2553        | 6.0   | 2892 | 0.9821   | 0.9589 | 0.1848          | 0.9608    | 0.9571 |
| 0.213         | 7.0   | 3374 | 0.9832   | 0.9613 | 0.1622          | 0.9616    | 0.9610 |
| 0.1819        | 8.0   | 3856 | 0.9858   | 0.9707 | 0.1430          | 0.9691    | 0.9722 |
| 0.16          | 9.0   | 4338 | 0.9871   | 0.9712 | 0.1307          | 0.9705    | 0.9719 |
| 0.1409        | 10.0  | 4820 | 0.9885   | 0.9749 | 0.1197          | 0.9734    | 0.9764 |
| 0.1295        | 11.0  | 5302 | 0.9893   | 0.9759 | 0.1120          | 0.9747    | 0.9771 |
| 0.1174        | 12.0  | 5784 | 0.9893   | 0.9763 | 0.1065          | 0.9744    | 0.9782 |
| 0.1085        | 13.0  | 6266 | 0.9896   | 0.9770 | 0.1005          | 0.9755    | 0.9785 |
| 0.1011        | 14.0  | 6748 | 0.9905   | 0.9794 | 0.0968          | 0.9786    | 0.9803 |
| 0.0954        | 15.0  | 7230 | 0.9910   | 0.9801 | 0.0941          | 0.9793    | 0.9810 |
| 0.0899        | 16.0  | 7712 | 0.9912   | 0.9807 | 0.0916          | 0.9796    | 0.9817 |
| 0.0866        | 17.0  | 8194 | 0.9917   | 0.9819 | 0.0893          | 0.9810    | 0.9828 |
| 0.0847        | 18.0  | 8676 | 0.9918   | 0.9824 | 0.0880          | 0.9814    | 0.9835 |
| 0.0814        | 19.0  | 9158 | 0.9918   | 0.9824 | 0.0870          | 0.9814    | 0.9835 |
| 0.0822        | 20.0  | 9640 | 0.9918   | 0.9824 | 0.0868          | 0.9814    | 0.9835 |


### Framework versions

- Transformers 4.45.2
- Pytorch 2.4.1+cu121
- Datasets 3.0.1
- Tokenizers 0.20.1
