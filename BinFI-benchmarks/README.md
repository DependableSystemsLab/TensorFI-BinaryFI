# Benchmarks in the BinFI paper

- This directory mainly contains the implementation of all the benchmarks evaluated in the SC19 paper. To run them and reproduce the results from the SC19 paper, you need to download the dataset and train the model. Then you can use BinFI to perform injection on the trained model.

- Note that you should make sure that all the models for different injection mode should be the same (i.e., the model which you perform exhaustive FI to validate the results from binFI should be the same as that for binFI). For more complex model, it is better to train the model and save the weights for further reuse.

- *To start with a simple example, you can run the LeNet-binFI.py and LeNet-allFI.py separately*, and compare the sdc rate measured by binFI with ground truth (by exhaustive FI) for one input. In this example, we have set up to always inject fault at the 2nd ADD operator in the LeNet model (2nd is already configured in the /TensorFI/injectFault.py - "randInstanceMap[op] = 2"). You need to config the default.yaml file under /LeNet-4/confFiles/ for the separate run. 

- The benchmarks and datasets are as follows. We also provide the link to the dataset.
    - 2 layer neural network - Mnist dataset (http://yann.lecun.com/exdb/mnist/)
    - LeNet-4 - Mnist dataset (http://yann.lecun.com/exdb/mnist/)
    - K-nearest neighbour - Survival dataset (https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival)
    - AlexNet - Cifar-10 (https://www.cs.toronto.edu/~kriz/cifar.html)
    - VGG16 - ImageNet (http://image-net.org/download)
    - VGG11 - German traffic sign (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
    - Comma.ai's steering model - real-world driving frame (https://github.com/SullyChen/driving-datasets)
    - Nvidia Dave steering model - real-world driving frame (https://github.com/SullyChen/driving-datasets)

- Note: Please make sure the configuration file (/confFiles/default.yaml) is properly configured. Detailed instruction is in the file.
