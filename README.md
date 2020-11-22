# Introduction
This repository contains the code used for "Unifying Input and Output Smoothing in Neural Machine Translation" (COLING2020). Our code is based on
[fairseq](https://github.com/pytorch/fairseq).

* **Architecture**
  + q_{src}, q_{tgt}: Input smoothing
  + q_{out}: Output/target smoothing
  + Φ: The parameter to control smoothing ratio and sampling method
  <div align=center>
  <img src="./images/arch.png"/ width="400px"> <br/>
  <img src="./images/formula.png"/ width="400px">
  </div>


