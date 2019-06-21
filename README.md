***BinFI: A binary fault injector for TensorFlow applications based on TensorFI (https://github.com/DependableSystemsLab/TensorFI)***
 
This repo is based on TensorFI, a fault injector for TensorFlow applications written in Python (https://github.com/DependableSystemsLab/TensorFI). It provides a new fault injection method - *binary fault injection*. Unlikely conventional random FI, BinFI performs a binary-search like FI and it is able to *efficiently identify the bits, where bit-flips can lead to SDCs.* We also provide exhaustive fault injection to perform injection on the whole state space (to validate the results by BinFI).

***How to run***

The major difference of this repo with TensorFI (https://github.com/DependableSystemsLab/TensorFI) in the /TensorFI/faultTypes.py module where we provide binary injection mode. So please follow the instruction in the TensorFI repo on how to install and run the tool.


***Prerequisite***

To apply BinFI on a ML model, you need to ensure that the computations within the ML model exhibit monotonicity property (i.e., for the same data point, larger value will result in larger outcome in the output; a typical example will be a MLP model). *Please see our SC19 paper for more details on how to analyze the monotonicity property of the ML computations*


***Paper***
BinFI: An Efficient Fault Injector for Safety-Critical Machine Learning Systems

