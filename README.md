***BinFI: A binary fault injector for TensorFlow applications based on TensorFI (https://github.com/DependableSystemsLab/TensorFI)***
 
This repo is based on TensorFI, a fault injector for TensorFlow applications written in Python. It provides a new fault injection method - *binary fault injection*. Unlikely conventional random FI, BinFI performs a binary-search like FI and it is able to *efficiently identify the bits, where bit-flips can lead to SDCs (e.g., image misclassification).* We also provide exhaustive fault injection to perform injection on the whole state space (to validate the results by BinFI).

***How to run***

The major difference of this repo with TensorFI is that we provide binary injection mode (in /TensorFI/faultTypes.py). You can use this feature by configuring the config file accordingly. The installation and step to run the tool is the same as that for TensorFI. We provide an example in /BinFI-benchmarks/LeNet-4/, where you can try running binary FI and exhaustive FI respectively.

***Prerequisite***

To apply BinFI on a ML model, you need to ensure that the computations within the ML model exhibit monotonicity property (i.e., for the same data point, larger value will result in larger outcome in the output; a typical example will be a MLP model). *Please see our SC19 paper for more details on how to analyze the monotonicity property of the ML computations*


***Paper***

Zitao Chen, Guanpeng Li, Karthik Pattabiraman, and Nathan DeBardeleben. 2019. BinFI: An Efficient Fault Injector for Safety-Critical Machine Learning Systems . In Proceeding of SC19, November 17-22, 2019, Denver, CO. ACM, New York, NY, USA, 14 pages.

