# How to reproduce results in the BinFI SC'19 paper (an example on LeNet)

1. Run LeNet-binFI.py (check config file is set properly), in this example, it will perform binary FI over two inputs. It also records the number of critical bits identified for each data. Remember that BinFI separates the bits of 0 and 1 for each data. This script will generate several files recording different results:
- lenet-binEach.csv: number of critical bits for each data, *critical bits for bits of 0 is in the odd-column; even-column for the bits of 1.*
- lenet-binFI.csv: overall SDC rate for each data, as well as the number of FI trials for binFI.
- data.csv: record the injected data (for validation)

2. Run LeNet-allFI.py (check config file is set properly), in this example, it will perform exhaustive FI over two inputs. This script will generate two files:
- lenet-seqEach.csv: FI result on each state space.
- lenet-seqFI.csv: overall SDC rate and number of FI trials.

3. Run examineResult.py. This will examine the recall and precision in identifying the critical bits by binFI, compared with the ground truth results. 

4. To measure the results for performing random FI, since we've already performed exhaustive FI and obtained the ground truth on the whole state space, so we can directly apply random profiling on the results by exhaustive FI. The examineResult.py script will also generate a file called randomFI-forDifferentTrials.csv, which records the number of critical bits for different number of trials when using random FI.

The above process illustrates the overall idea on validating the results by BinFI. To run BinFI on other models, you should change some parameters (the file name containing your results) in examineResult.py accordingly.

