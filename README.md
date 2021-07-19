# :fire: SNGP-BERT (Unofficial) :fire:
This is reimplementation of **"Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness"** in Pytorch.

The codes are based on [official repo (Tensorflow)](https://github.com/google/uncertainty-baselines/blob/main/baselines/clinc_intent/sngp.py) and [huggingface](https://huggingface.co/).

Original Paper : [Link](https://arxiv.org/pdf/2006.10108.pdf)

## Installation :coffee:

Training environment : Ubuntu 18.04, python 3.6
```bash
pip3 install torch torchvision torchaudio
pip install scikit-learn
```

Download `bert-base-uncased` checkpoint from [hugginface-ckpt](https://huggingface.co/bert-base-uncased/tree/main)  
Download `bert-base-uncased` vocab file from [hugginface-vocab](https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt)  
Download CLINC OOS intent detection benchmark dataset from [tensorflow-dataset](https://github.com/jereliu/datasets/raw/master/clinc_oos.zip)

The downloaded files' directory should be:

```bash
SNGP-BERT
ㄴckpt
  ㄴbert-base-uncased-pytorch_model.bin
ㄴdataset
  ㄴclinc_oos
    ㄴtrain.csv
    ㄴval.csv
    ㄴtest.csv
    ㄴtest_ood.csv
  ㄴvocab
    ㄴbert-base-uncased-vocab.txt
ㄴmodels
...
```


## Dataset Info :book:

In their paper, the authors conducted OOD experiment for NLP using CLINC OOS intent detection benchmark dataset, the OOS dataset contains data for 150 in-domain services with 150 training
sentences in each domain, and also 1500 natural out-of-domain utterances.
You can download the dataset at [Link](https://github.com/jereliu/datasets/raw/master/clinc_oos.zip).

Original dataset paper, and Github : [Paper Link](https://aclanthology.org/D19-1131/), [Git Link](https://github.com/clinc/oos-eval)

## Run :star2:

#### Train
```bash
python main.py --train_or_test train --method sngp --device gpu --gpu 0
```

#### Test

```bash
python main.py --train_or_test test --method sngp --device gpu --gpu 0
```

## Results :sparkles:

Results for `SNGP-BERT` on CLINC OOS.  
**NOTE** : Depending on the random seed, the result may be slightly different.

| Version | ACC | AUROC | AUPRC |
| --- | --- | --- | --- |
| Paper (Tensorflow) | 96.6 | 0.969 | 0.880 |
| Pytorch (batch size = 256) | 96.1 | 0.974 | 0.900 |
| Pytorch (batch size = 64) | 95.9 | 0.972 | 0.894 |


## References

[1] https://github.com/google/uncertainty-baselines/blob/main/baselines/clinc_intent/sngp.py  
[2] https://huggingface.co/  
[3] https://github.com/google/edward2