# Introduction
This repository contains the code used for "Unifying Input and Output Smoothing in Neural Machine Translation" (COLING2020). Our code is based on
[fairseq](https://github.com/pytorch/fairseq).

* **Architecture**
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


