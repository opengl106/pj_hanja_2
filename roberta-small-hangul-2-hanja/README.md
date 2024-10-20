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
- Accuracy: 0.9956
- F1: 0.9902
- Loss: 0.0352
- Precision: 0.9894
- Recall: 0.9911

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
- num_epochs: 150

### Training results

| Training Loss | Epoch | Step  | Accuracy | F1     | Validation Loss | Precision | Recall |
|:-------------:|:-----:|:-----:|:--------:|:------:|:---------------:|:---------:|:------:|
| No log        | 1.0   | 482   | 0.8993   | 0.4607 | 0.8157          | 0.6392    | 0.3602 |
| 1.7229        | 2.0   | 964   | 0.9606   | 0.8537 | 0.4586          | 0.8728    | 0.8354 |
| 0.6279        | 3.0   | 1446  | 0.9701   | 0.9159 | 0.3291          | 0.9244    | 0.9075 |
| 0.418         | 4.0   | 1928  | 0.9761   | 0.9426 | 0.2596          | 0.9439    | 0.9413 |
| 0.316         | 5.0   | 2410  | 0.9797   | 0.9530 | 0.2159          | 0.9545    | 0.9515 |
| 0.2553        | 6.0   | 2892  | 0.9821   | 0.9589 | 0.1848          | 0.9608    | 0.9571 |
| 0.213         | 7.0   | 3374  | 0.9832   | 0.9613 | 0.1622          | 0.9616    | 0.9610 |
| 0.1819        | 8.0   | 3856  | 0.9858   | 0.9707 | 0.1430          | 0.9691    | 0.9722 |
| 0.16          | 9.0   | 4338  | 0.9871   | 0.9712 | 0.1307          | 0.9705    | 0.9719 |
| 0.1409        | 10.0  | 4820  | 0.9885   | 0.9749 | 0.1197          | 0.9734    | 0.9764 |
| 0.1295        | 11.0  | 5302  | 0.9893   | 0.9759 | 0.1120          | 0.9747    | 0.9771 |
| 0.1174        | 12.0  | 5784  | 0.9893   | 0.9763 | 0.1065          | 0.9744    | 0.9782 |
| 0.1085        | 13.0  | 6266  | 0.9896   | 0.9770 | 0.1005          | 0.9755    | 0.9785 |
| 0.1011        | 14.0  | 6748  | 0.9905   | 0.9794 | 0.0968          | 0.9786    | 0.9803 |
| 0.0954        | 15.0  | 7230  | 0.9910   | 0.9801 | 0.0941          | 0.9793    | 0.9810 |
| 0.0899        | 16.0  | 7712  | 0.9912   | 0.9807 | 0.0916          | 0.9796    | 0.9817 |
| 0.0866        | 17.0  | 8194  | 0.9917   | 0.9819 | 0.0893          | 0.9810    | 0.9828 |
| 0.0847        | 18.0  | 8676  | 0.9918   | 0.9824 | 0.0880          | 0.9814    | 0.9835 |
| 0.0814        | 19.0  | 9158  | 0.9918   | 0.9824 | 0.0870          | 0.9814    | 0.9835 |
| 0.0822        | 20.0  | 9640  | 0.9918   | 0.9824 | 0.0868          | 0.9814    | 0.9835 |
| 0.0802        | 21.0  | 10122 | 0.9929   | 0.9922 | 0.0617          | 0.9924    | 0.9920 |
| 0.0734        | 22.0  | 10604 | 0.9932   | 0.9907 | 0.0562          | 0.9903    | 0.9911 |
| 0.0665        | 23.0  | 11086 | 0.9942   | 0.9933 | 0.0517          | 0.9937    | 0.9929 |
| 0.0582        | 24.0  | 11568 | 0.9939   | 0.9924 | 0.0485          | 0.9922    | 0.9926 |
| 0.0526        | 25.0  | 12050 | 0.9944   | 0.9926 | 0.0452          | 0.9926    | 0.9926 |
| 0.0468        | 26.0  | 12532 | 0.9947   | 0.9933 | 0.0423          | 0.9926    | 0.9940 |
| 0.0419        | 27.0  | 13014 | 0.9960   | 0.9945 | 0.0299          | 0.9943    | 0.9947 |
| 0.0419        | 28.0  | 13496 | 0.9959   | 0.9934 | 0.0352          | 0.9929    | 0.9940 |
| 0.0382        | 29.0  | 13978 | 0.9964   | 0.9961 | 0.0342          | 0.9954    | 0.9968 |
| 0.0346        | 30.0  | 14460 | 0.9955   | 0.9940 | 0.0335          | 0.9933    | 0.9947 |
| 0.0313        | 31.0  | 14942 | 0.9957   | 0.9938 | 0.0318          | 0.9929    | 0.9947 |
| 0.0286        | 32.0  | 15424 | 0.9958   | 0.9943 | 0.0310          | 0.9936    | 0.9950 |
| 0.026         | 33.0  | 15906 | 0.9961   | 0.9950 | 0.0304          | 0.9943    | 0.9957 |
| 0.0234        | 34.0  | 16388 | 0.9960   | 0.9940 | 0.0292          | 0.9933    | 0.9947 |
| 0.0217        | 35.0  | 16870 | 0.9957   | 0.9941 | 0.0279          | 0.9933    | 0.9950 |
| 0.0198        | 36.0  | 17352 | 0.9958   | 0.9927 | 0.0272          | 0.9922    | 0.9933 |
| 0.0178        | 37.0  | 17834 | 0.9959   | 0.9933 | 0.0264          | 0.9926    | 0.9940 |
| 0.0167        | 38.0  | 18316 | 0.9958   | 0.9931 | 0.0266          | 0.9922    | 0.9940 |
| 0.0149        | 39.0  | 18798 | 0.9960   | 0.9931 | 0.0262          | 0.9922    | 0.9940 |
| 0.0137        | 40.0  | 19280 | 0.9956   | 0.9925 | 0.0255          | 0.9918    | 0.9933 |
| 0.013         | 41.0  | 19762 | 0.9958   | 0.9927 | 0.0253          | 0.9918    | 0.9936 |
| 0.0115        | 42.0  | 20244 | 0.9959   | 0.9918 | 0.0250          | 0.9915    | 0.9922 |
| 0.0107        | 43.0  | 20726 | 0.9957   | 0.9920 | 0.0258          | 0.9911    | 0.9929 |
| 0.0098        | 44.0  | 21208 | 0.9959   | 0.9931 | 0.0248          | 0.9922    | 0.9940 |
| 0.009         | 45.0  | 21690 | 0.9960   | 0.9920 | 0.0254          | 0.9911    | 0.9929 |
| 0.0081        | 46.0  | 22172 | 0.9959   | 0.9925 | 0.0258          | 0.9918    | 0.9933 |
| 0.0075        | 47.0  | 22654 | 0.9956   | 0.9915 | 0.0251          | 0.9904    | 0.9925 |
| 0.0069        | 48.0  | 23136 | 0.9956   | 0.9913 | 0.0256          | 0.9904    | 0.9922 |
| 0.0063        | 49.0  | 23618 | 0.9955   | 0.9906 | 0.0262          | 0.9894    | 0.9918 |
| 0.0055        | 50.0  | 24100 | 0.9959   | 0.9913 | 0.0254          | 0.9904    | 0.9922 |
| 0.0052        | 51.0  | 24582 | 0.9956   | 0.9911 | 0.0255          | 0.9901    | 0.9922 |
| 0.0048        | 52.0  | 25064 | 0.9958   | 0.9910 | 0.0256          | 0.9901    | 0.9918 |
| 0.0044        | 53.0  | 25546 | 0.9954   | 0.9892 | 0.0276          | 0.9883    | 0.9901 |
| 0.0039        | 54.0  | 26028 | 0.9955   | 0.9897 | 0.0271          | 0.9890    | 0.9904 |
| 0.0037        | 55.0  | 26510 | 0.9957   | 0.9897 | 0.0275          | 0.9887    | 0.9908 |
| 0.0037        | 56.0  | 26992 | 0.9957   | 0.9910 | 0.0273          | 0.9901    | 0.9918 |
| 0.0032        | 57.0  | 27474 | 0.9960   | 0.9910 | 0.0270          | 0.9901    | 0.9918 |
| 0.003         | 58.0  | 27956 | 0.9955   | 0.9904 | 0.0284          | 0.9890    | 0.9918 |
| 0.0027        | 59.0  | 28438 | 0.9956   | 0.9904 | 0.0287          | 0.9890    | 0.9918 |
| 0.0024        | 60.0  | 28920 | 0.9955   | 0.9892 | 0.0290          | 0.9876    | 0.9908 |
| 0.0022        | 61.0  | 29402 | 0.9958   | 0.9910 | 0.0286          | 0.9901    | 0.9918 |
| 0.0021        | 62.0  | 29884 | 0.9959   | 0.9913 | 0.0286          | 0.9904    | 0.9922 |
| 0.0019        | 63.0  | 30366 | 0.9954   | 0.9897 | 0.0325          | 0.9883    | 0.9911 |
| 0.0017        | 64.0  | 30848 | 0.9957   | 0.9906 | 0.0295          | 0.9897    | 0.9915 |
| 0.0015        | 65.0  | 31330 | 0.9958   | 0.9915 | 0.0289          | 0.9908    | 0.9922 |
| 0.0014        | 66.0  | 31812 | 0.9957   | 0.9911 | 0.0303          | 0.9901    | 0.9922 |
| 0.0012        | 67.0  | 32294 | 0.9956   | 0.9904 | 0.0306          | 0.9890    | 0.9918 |
| 0.0012        | 68.0  | 32776 | 0.9955   | 0.9899 | 0.0312          | 0.9887    | 0.9911 |
| 0.0011        | 69.0  | 33258 | 0.9956   | 0.9897 | 0.0310          | 0.9883    | 0.9911 |
| 0.001         | 70.0  | 33740 | 0.9957   | 0.9895 | 0.0309          | 0.9883    | 0.9908 |
| 0.001         | 71.0  | 34222 | 0.9957   | 0.9901 | 0.0322          | 0.9897    | 0.9904 |
| 0.0008        | 72.0  | 34704 | 0.9958   | 0.9901 | 0.0323          | 0.9894    | 0.9908 |
| 0.0008        | 73.0  | 35186 | 0.9956   | 0.9897 | 0.0312          | 0.9890    | 0.9904 |
| 0.0007        | 74.0  | 35668 | 0.9957   | 0.9901 | 0.0327          | 0.9894    | 0.9908 |
| 0.0007        | 75.0  | 36150 | 0.9958   | 0.9911 | 0.0315          | 0.9904    | 0.9918 |
| 0.0007        | 76.0  | 36632 | 0.9957   | 0.9911 | 0.0318          | 0.9901    | 0.9922 |
| 0.0006        | 77.0  | 37114 | 0.9958   | 0.9911 | 0.0314          | 0.9901    | 0.9922 |
| 0.0006        | 78.0  | 37596 | 0.9956   | 0.9904 | 0.0325          | 0.9890    | 0.9918 |
| 0.0005        | 79.0  | 38078 | 0.9955   | 0.9894 | 0.0318          | 0.9880    | 0.9908 |
| 0.0005        | 80.0  | 38560 | 0.9957   | 0.9904 | 0.0315          | 0.9901    | 0.9908 |
| 0.0004        | 81.0  | 39042 | 0.9958   | 0.9897 | 0.0321          | 0.9897    | 0.9897 |
| 0.0004        | 82.0  | 39524 | 0.9952   | 0.9895 | 0.0340          | 0.9883    | 0.9908 |
| 0.0005        | 83.0  | 40006 | 0.9956   | 0.9915 | 0.0317          | 0.9908    | 0.9922 |
| 0.0005        | 84.0  | 40488 | 0.9955   | 0.9901 | 0.0324          | 0.9887    | 0.9915 |
| 0.0004        | 85.0  | 40970 | 0.9956   | 0.9910 | 0.0320          | 0.9901    | 0.9918 |
| 0.0003        | 86.0  | 41452 | 0.9957   | 0.9918 | 0.0324          | 0.9911    | 0.9925 |
| 0.0003        | 87.0  | 41934 | 0.9959   | 0.9915 | 0.0308          | 0.9908    | 0.9922 |
| 0.0003        | 88.0  | 42416 | 0.9956   | 0.9913 | 0.0337          | 0.9904    | 0.9922 |
| 0.0003        | 89.0  | 42898 | 0.9956   | 0.9913 | 0.0330          | 0.9904    | 0.9922 |
| 0.0003        | 90.0  | 43380 | 0.9956   | 0.9890 | 0.0330          | 0.9876    | 0.9904 |
| 0.0003        | 91.0  | 43862 | 0.9957   | 0.9911 | 0.0341          | 0.9901    | 0.9922 |
| 0.0003        | 92.0  | 44344 | 0.9956   | 0.9906 | 0.0337          | 0.9894    | 0.9918 |
| 0.0002        | 93.0  | 44826 | 0.9956   | 0.9906 | 0.0343          | 0.9897    | 0.9915 |
| 0.0003        | 94.0  | 45308 | 0.9957   | 0.9910 | 0.0336          | 0.9901    | 0.9918 |
| 0.0002        | 95.0  | 45790 | 0.9954   | 0.9901 | 0.0355          | 0.9887    | 0.9915 |
| 0.0002        | 96.0  | 46272 | 0.9958   | 0.9904 | 0.0326          | 0.9894    | 0.9915 |
| 0.0002        | 97.0  | 46754 | 0.9959   | 0.9901 | 0.0334          | 0.9894    | 0.9908 |
| 0.0002        | 98.0  | 47236 | 0.9959   | 0.9908 | 0.0337          | 0.9897    | 0.9918 |
| 0.0002        | 99.0  | 47718 | 0.9958   | 0.9908 | 0.0334          | 0.9897    | 0.9918 |
| 0.0002        | 100.0 | 48200 | 0.9957   | 0.9902 | 0.0347          | 0.9890    | 0.9915 |
| 0.0002        | 101.0 | 48682 | 0.9953   | 0.9894 | 0.0379          | 0.9873    | 0.9915 |
| 0.0002        | 102.0 | 49164 | 0.9957   | 0.9902 | 0.0340          | 0.9890    | 0.9915 |
| 0.0002        | 103.0 | 49646 | 0.9956   | 0.9894 | 0.0336          | 0.9890    | 0.9897 |
| 0.0001        | 104.0 | 50128 | 0.9954   | 0.9911 | 0.0362          | 0.9901    | 0.9922 |
| 0.0002        | 105.0 | 50610 | 0.9956   | 0.9902 | 0.0339          | 0.9890    | 0.9915 |
| 0.0002        | 106.0 | 51092 | 0.9957   | 0.9904 | 0.0339          | 0.9901    | 0.9908 |
| 0.0002        | 107.0 | 51574 | 0.9957   | 0.9897 | 0.0341          | 0.9900    | 0.9893 |
| 0.0002        | 108.0 | 52056 | 0.9955   | 0.9897 | 0.0350          | 0.9883    | 0.9911 |
| 0.0001        | 109.0 | 52538 | 0.9956   | 0.9910 | 0.0334          | 0.9901    | 0.9918 |
| 0.0001        | 110.0 | 53020 | 0.9954   | 0.9897 | 0.0364          | 0.9887    | 0.9908 |
| 0.0001        | 111.0 | 53502 | 0.9956   | 0.9886 | 0.0340          | 0.9879    | 0.9893 |
| 0.0001        | 112.0 | 53984 | 0.9955   | 0.9895 | 0.0346          | 0.9880    | 0.9911 |
| 0.0001        | 113.0 | 54466 | 0.9954   | 0.9897 | 0.0348          | 0.9887    | 0.9908 |
| 0.0001        | 114.0 | 54948 | 0.9956   | 0.9906 | 0.0347          | 0.9894    | 0.9918 |
| 0.0001        | 115.0 | 55430 | 0.9956   | 0.9899 | 0.0342          | 0.9890    | 0.9908 |
| 0.0001        | 116.0 | 55912 | 0.9957   | 0.9908 | 0.0344          | 0.9901    | 0.9915 |
| 0.0001        | 117.0 | 56394 | 0.9956   | 0.9895 | 0.0340          | 0.9887    | 0.9904 |
| 0.0001        | 118.0 | 56876 | 0.9955   | 0.9895 | 0.0347          | 0.9880    | 0.9911 |
| 0.0001        | 119.0 | 57358 | 0.9955   | 0.9901 | 0.0349          | 0.9887    | 0.9915 |
| 0.0001        | 120.0 | 57840 | 0.9955   | 0.9892 | 0.0351          | 0.9880    | 0.9904 |
| 0.0001        | 121.0 | 58322 | 0.9955   | 0.9899 | 0.0359          | 0.9883    | 0.9915 |
| 0.0001        | 122.0 | 58804 | 0.9955   | 0.9890 | 0.0365          | 0.9873    | 0.9908 |
| 0.0001        | 123.0 | 59286 | 0.9955   | 0.9883 | 0.0350          | 0.9869    | 0.9897 |
| 0.0001        | 124.0 | 59768 | 0.9955   | 0.9890 | 0.0344          | 0.9880    | 0.9901 |
| 0.0001        | 125.0 | 60250 | 0.9956   | 0.9890 | 0.0352          | 0.9897    | 0.9883 |
| 0.0001        | 126.0 | 60732 | 0.9953   | 0.9887 | 0.0359          | 0.9869    | 0.9904 |
| 0.0001        | 127.0 | 61214 | 0.9952   | 0.9876 | 0.0360          | 0.9862    | 0.9890 |
| 0.0001        | 128.0 | 61696 | 0.9953   | 0.9888 | 0.0357          | 0.9869    | 0.9908 |
| 0.0001        | 129.0 | 62178 | 0.9954   | 0.9888 | 0.0363          | 0.9869    | 0.9908 |
| 0.0001        | 130.0 | 62660 | 0.9954   | 0.9894 | 0.0360          | 0.9880    | 0.9908 |
| 0.0001        | 131.0 | 63142 | 0.9956   | 0.9890 | 0.0360          | 0.9883    | 0.9897 |
| 0.0001        | 132.0 | 63624 | 0.9954   | 0.9885 | 0.0362          | 0.9869    | 0.9901 |
| 0.0001        | 133.0 | 64106 | 0.9955   | 0.9888 | 0.0356          | 0.9876    | 0.9901 |
| 0.0001        | 134.0 | 64588 | 0.9954   | 0.9894 | 0.0367          | 0.9876    | 0.9911 |
| 0.0001        | 135.0 | 65070 | 0.9954   | 0.9894 | 0.0364          | 0.9876    | 0.9911 |
| 0.0001        | 136.0 | 65552 | 0.9954   | 0.9890 | 0.0363          | 0.9876    | 0.9904 |
| 0.0           | 137.0 | 66034 | 0.9954   | 0.9894 | 0.0369          | 0.9876    | 0.9911 |
| 0.0001        | 138.0 | 66516 | 0.9954   | 0.9894 | 0.0369          | 0.9876    | 0.9911 |
| 0.0001        | 139.0 | 66998 | 0.9956   | 0.9902 | 0.0358          | 0.9894    | 0.9911 |
| 0.0001        | 140.0 | 67480 | 0.9956   | 0.9901 | 0.0360          | 0.9887    | 0.9915 |
| 0.0001        | 141.0 | 67962 | 0.9957   | 0.9902 | 0.0357          | 0.9894    | 0.9911 |
| 0.0001        | 142.0 | 68444 | 0.9957   | 0.9902 | 0.0355          | 0.9894    | 0.9911 |
| 0.0001        | 143.0 | 68926 | 0.9957   | 0.9888 | 0.0349          | 0.9883    | 0.9893 |
| 0.0           | 144.0 | 69408 | 0.9956   | 0.9895 | 0.0357          | 0.9883    | 0.9908 |
| 0.0           | 145.0 | 69890 | 0.9955   | 0.9899 | 0.0361          | 0.9883    | 0.9915 |
| 0.0001        | 146.0 | 70372 | 0.9956   | 0.9902 | 0.0350          | 0.9894    | 0.9911 |
| 0.0001        | 147.0 | 70854 | 0.9956   | 0.9906 | 0.0354          | 0.9894    | 0.9918 |
| 0.0           | 148.0 | 71336 | 0.9956   | 0.9902 | 0.0351          | 0.9894    | 0.9911 |
| 0.0           | 149.0 | 71818 | 0.9956   | 0.9902 | 0.0352          | 0.9894    | 0.9911 |
| 0.0           | 150.0 | 72300 | 0.9956   | 0.9902 | 0.0352          | 0.9894    | 0.9911 |


### Framework versions

- Transformers 4.45.2
- Pytorch 2.4.1+cu121
- Datasets 3.0.1
- Tokenizers 0.20.1
