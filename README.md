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
### Data Preprocessing
Following the standard [fairseq](https://github.com/pytorch/fairseq) data preprocessing, you can obtain binary translation dataset. For example:
  * IWSLT14 German to English
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
  * WMT14 English to German
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

### Training
There are some flags you can set:

'''
p=0.2
s=1
python train.py \
                                /work/smt2/bliao/dataset/data-bin/nl_en \
                                --share-all-embeddings \
                                --log-format simple \
                                --arch transformer_da_bert_iwslt_de_en \
                                --task translation_da \
                                --srcda \
                                --srcda-percentage $p \
                                --srcda-choice uniform \
                                --srcda-smooth $s \
                                --tgtda \
                                --tgtda-percentage $p \
                                --tgtda-choice uniform \
                                --tgtda-smooth $s \
                                --select-choice uniform \
                                --optimizer adam \
                                --lr 0.0005  -s nl -t en \
                                --label-smoothing 0.1 \
                                --dropout 0.3 \
                                --max-tokens 4096 \
                                --min-lr '1e-09' \
                                --seed $n \
                                --lr-scheduler inverse_sqrt \
                                --weight-decay 0.0001 \
                                --criterion label_smoothed_cross_entropy \
                                --warmup-updates 4000 \
                                --warmup-init-lr '1e-07' \
                                --adam-betas '(0.9, 0.98)' \
                                --clip-norm 0.0 \
                                --keep-last-epochs 5 \
                                --patience 10 \
                                --save-dir $checkpoint_path 
'''



