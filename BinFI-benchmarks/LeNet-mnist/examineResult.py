import csv
import pandas
import numpy as np 


global totalCriticalBit

totalCriticalBit = []

# convert the injected data into binary format
def binConv(file):

	def float2bin(number, decLength = 10): 
		# convert float data into binary expression
		# we consider fixed points data type

		# split integer and decimal part into seperate variables  
		integer, decimal = str(number).split(".") 
		# convert integer and decimal part into integer  
		integer = int(integer)  
		# Convert the integer part into binary form. 
		res = bin(integer)[2:] + "."		# strip the first binary label "0b"

		# 21 fixed digits for integer digit, 22 because of the decimal point "."
		res = res.zfill(22)
		# Convert the value passed as parameter to it's decimal representation 
		def decimalConverter(decimal): 
			decimal = '0' + '.' + decimal 
			return float(decimal)

		# iterate times = length of binary decimal part
		for x in range(decLength): 
			# Multiply the decimal value by 2 and seperate the integer and decimal parts 
			# formating the digits so that it would not be expressed by scientific notation
			integer, decimal = format( (decimalConverter(decimal)) * 2, '.10f' ).split(".")    
			res += integer 

		return res 


	writes = open("binaryInjectedData.csv", "w")	# write the injected data in binary expression

	numOfInjectedData = 0 # this is the number of data within the operation, e.g., (2, 2, 2) = 8
	numOfInjectedInput = 0 # this is the number of input we've evaluated, e.g., 10 in BinFI paper.
	# read the original injected data
	res = open(file , "r")
	reader = csv.reader(res)
	for each in reader:
		each = each[:-1] 
 		numOfInjectedInput += 1

		for e in each:
			numOfInjectedData += 1
			# convert into binary expression
			data = abs(float(e))
			bins = float2bin(data)
			for bit in bins:
				if(bit == '.'):
					continue
				writes.write(bit + ",")
		writes.write("\n")

	return numOfInjectedInput, numOfInjectedData/numOfInjectedInput

# map the binFI results into the per-bit SDC case in a file, to be compared with all FI
# E.g., for a data with 31-bit, how many of them are critical bits. Each data is expressed in 31 binary bit
def getPerBit_SDC_byBinFI(numOfInjectedInput, numOfInjectedData, BinFIresFile):
	# read the results from binFI. The odd-column is the critical bits from the bits of 0, even- for the bits of 1
	# numOfInjectedData * 2 because we consider 0 and 1 bits for each data.
	binRes = np.zeros((numOfInjectedInput,numOfInjectedData*2))
	binFI = open(BinFIresFile, "r")
	reads = csv.reader(binFI)
	index = 0
	for each in reads: 
		each = each[:-1]
 
		for i in range(len(each)): 
			binRes[index][i] = int(each[i])
		index+=1

	writePerBitSDC = open("binFIres-intoPerBit.csv", "w")	# write the per-bit SDC results into a file 
 
	binData = open("binaryInjectedData.csv", "r")	# read the injected values in binary
	reader = csv.reader(binData)
	dataIndex = 0
	for each in reader: 
		for index in range(numOfInjectedData): 
			e = each[index*31 : (index+1)*31]
			bit0 = binRes[dataIndex][index*2]	# number of critical bits for the bits of 0 identified by binFI
			bit1 = binRes[dataIndex][index*2+1] # number of critical bits for the bits of 1 identified by binFI
 
			for i in e: 
				if(i == '0'):
					if(bit0 > 0):
						# write "0" means this is a critical bits
						writePerBitSDC.write("0" + ",") 
						bit0 -= 1
					else:
						writePerBitSDC.write("1" + ",") 
				elif(i == '1'):
					if(bit1 > 0):
						# write "0" means this is a critical bits
						writePerBitSDC.write("0" + ",") 
						bit1 -= 1
					else:
						writePerBitSDC.write("1" + ",")  
		dataIndex += 1
		writePerBitSDC.write("\n")

# compare the results from binFI and exhaustive FI, bit-by-bit
def validateBinFIres(numOfInjectedInput, numOfInjectedData, exhaustiveFIresFile):
	# read the results from binFI
	binFI = open('binFIres-intoPerBit.csv', "r")
	reader = csv.reader(binFI)

	binRes = np.zeros((numOfInjectedInput, numOfInjectedData*31))
	index = 0
	for each in reader:
		tmp = each[:-1]
		tmp = np.asarray(tmp)
		tmp = tmp.astype(int)
		binRes[index] = tmp
		index +=1

	# read the results from exhaustive FI
	allFI = open(exhaustiveFIresFile, "r")
	reader = csv.reader(allFI)

	allRes = np.zeros((numOfInjectedInput, numOfInjectedData*31))
	index = 0
	for each in reader:
		tmp = each[:-1]
		tmp = np.asarray(tmp)
		tmp = tmp.astype(int)
		allRes[index] = tmp
		index +=1
 
	binFI_trial = readBinFI_trial("lenet-binFI.csv")
	for i in range(numOfInjectedInput):
		fp = 0 
		fn = 0
		totalBit = 0
		for j in range(numOfInjectedData*31): 
			if(allRes[i][j]==0):
				totalBit+=1


			if(allRes[i][j] == 1 and binRes[i][j] == 0):
				fp += 1
			elif(allRes[i][j] == 0 and binRes[i][j] == 1):
				fn+=1
		
		print("=============== new data entry for binary FI results: ===============")
		print('BinFI recall rate: %.2f, BinFI precision: %.2f, total critical bits: %d, FI trial for BinFI: %d, '  
				%(  float(totalBit-fn)/totalBit, 1-float(fp)/totalBit, totalBit, binFI_trial[i] ))

		global totalCriticalBit
		totalCriticalBit.append(totalBit)


# read the num of trial for binary FI
def readBinFI_trial(fileName):
	binFIRes = open(fileName, "r")
	data = csv.reader(binFIRes)
	trial = []
	for each in data:
		trial.append( int(each[1]) )
	return trial	

# perform randomFI, since we've the ground truth from exhaustive FI already,
# we can "simulate" random FI on the ground truth table
# this function is to collect the cumulative critical bits identified for given FI trials
def ranFI(numOfInjectedData, canDuplicate, exhaustiveFIresFile):

	global totalCriticalBit 

	binFI_trial = readBinFI_trial("lenet-binFI.csv") 
	
	groundTruthRes = open(exhaustiveFIresFile, "r")
	data = csv.reader(groundTruthRes)

	ranRes = open("randomFI-forDifferentTrials.csv", "w")	# write the results for random FI

	cnt = 0 
	for each in data: 
		print("=============== new data entry for random FI results: ===============")
		visitedInd = [] # index of injected bit (random injection)

		allRes = each[:-1] 
		allRes = np.asarray(allRes)
		allRes = allRes.astype(float)
  
		criticalBit = 0.
		for i in range(numOfInjectedData*31):
			# random index of the injected data
			numInd = np.random.randint(low=0 , high = numOfInjectedData)
			# random index of the injected bit
			bitInd = np.random.randint(low=0 , high = 31)
   
			if(allRes[ numInd*31 + bitInd ] == 0. and (canDuplicate or ((numInd*31 + bitInd) not in visitedInd))): 
				criticalBit+=1
			visitedInd.append(numInd*31 + bitInd)
			
			ranRes.write(`criticalBit` + ",")

			if (i+1)==numOfInjectedData*31 or (i+1) == numOfInjectedData*31/2 or (i+1) == numOfInjectedData*31/4 or (i+1) == binFI_trial[cnt] or (i+1) == binFI_trial[cnt]/2 or (i+1) == binFI_trial[cnt]/4:
			    	print "num of random FI trial: ", i, " Recall rate: ", criticalBit / totalCriticalBit[cnt] 
				a = 1
		
		ranRes.write("\n") 
		cnt+=1 
 

 
"NOTE: this script is written for the sample LeNet test, " 
"you can check the results of BinFI on different models, by choosing different FileName parameters for you results"

originalInjectedData = 'data.csv'	# this is the file containg the original injected data.

# convert the injected data into binary expression
numOfInjectedInput, numOfInjectedData =  binConv(originalInjectedData)
# map the binFI results into per-bit results for each data
getPerBit_SDC_byBinFI(numOfInjectedInput, numOfInjectedData, BinFIresFile="lenet-binEach.csv")
# compare binFI with the exhaustive FI (ground truth)
validateBinFIres(numOfInjectedInput, numOfInjectedData, exhaustiveFIresFile="lenet-seqEach.csv")


# get the number of critical bits collected by random FI, in different FI trials
ranFI(numOfInjectedData, canDuplicate = False, exhaustiveFIresFile="lenet-seqEach.csv")







