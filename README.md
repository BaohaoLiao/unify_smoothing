# Introduction
This repository contains the code used for "Unifying Input and Output Smoothing in Neural Machine Translation" (COLING2020). Our code is based on
[fairseq](https://github.com/pytorch/fairseq).

### Architecture
  + q_{src}, q_{tgt}: Input smoothing
  + q_{out}: Output/target smoothing
  + Φ: The parameter to control smoothing weight and sampling method
  + m: Smoothing weight
  <div align=center>
  <img src="./images/arch.png"/ width="400px"> <br/>
  <img src="./images/formula.png"/ width="400px">
  </div>
  
# Requirements and Installation
* PyTorch version >= 1.5.0
* Python version >= 3.6
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/BaohaoLiao/unify_smoothing.git
cd unify_smoothing
pip install --editable ./
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library.

# Getting Started
* **Data Preprocessing**
Following the standard [fairseq](https://github.com/pytorch/fairseq) data preprocessing, you can obtain binary translation dataset. For example:
  + IWSLT14 German to English
```
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```
  + WMT14 English to German
```
  # Download and prepare the data
cd examples/translation/
# WMT'17 data:
bash prepare-wmt14en2de.sh
# or to use WMT'14 data:
# bash prepare-wmt14en2de.sh --icml17
cd ../..

# Binarize the dataset
TEXT=examples/translation/wmt17_en_de
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
  ```


