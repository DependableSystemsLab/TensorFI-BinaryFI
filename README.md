***BinFI: A binary fault injector for TensorFlow applications based on TensorFI (https://github.com/DependableSystemsLab/TensorFI)***
 
This repo is based on TensorFI, a fault injector for TensorFlow applications written in Python. It provides a faster fault injection method - *binary fault injection*. Unlike conventional random FI, BinFI performs a binary-search like FI and it is able to *efficiently identify SDC-causing bits, that is, the bits lead to SDCs(e.g., image misclassification) if flipped.* We also provide exhaustive fault injection to inject faults into the whole state space to evaluate the efficacy of BinFI.

***How to run***

The major difference of this repo with TensorFI is that we provide binary injection mode (in /TensorFI/faultTypes.py). You can use this feature by configuring the config file accordingly. The installation and step to run the tool is the same as that for TensorFI. You can install TensorFI-BinaryFI by running the Install.sh (which will install TensorFI-BinaryFI in a virtual environment) OR by pip install (pip install TensorFI-BinaryFI).

We provide a detailed example in /BinFI-benchmarks/LeNet-4/ (with instructions), where you can try running binary FI and exhaustive FI respectively. NOTE: Running these two FI approaches require modification to the TF program and config file. You can modify your own program following the LeNet example. A general way for using BinFI is as follows:

1. Configure the TF program and config file for binary FI (e.g., see line 380 to the end in LeNet-binFI.py). And then performing FI and collecting the stats, i.e., num of critical bits identified for each data; the data to be injected.

2. Configure the TF program for exhaustive FI (e.g., see line 380 to the end in LeNet-allFI.py), it will collect the FI result on each state space.

3. Examine the results by binary FI and exhaustive FI. The script for the examination is provided in /BinFI-benchmarks/LeNet-4/examineReesult.py.



***Prerequisite***

To apply BinFI on a ML model, you need to ensure that the computations within the ML model exhibit monotonicity property (i.e., for the same data point, larger value will result in larger outcome in the output; a typical example will be a MLP model). *Please see our SC19 paper for more details on how to analyze the monotonicity property of the ML computations*


***Paper***

Zitao Chen, Guanpeng Li, Karthik Pattabiraman, and Nathan DeBardeleben. 2019. BinFI: An Efficient Fault Injector for Safety-Critical Machine Learning Systems . In The International Conference for High Performance Computing, Networking, Storage, and Analysis (SC ’19), November 17–22, 2019, Denver, CO, USA. ACM, New York, NY, USA, 14 pages. https://doi.org/10.1145/3295500.3356177

