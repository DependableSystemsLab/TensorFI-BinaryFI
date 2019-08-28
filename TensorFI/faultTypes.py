# These are the list of fault injection functions for different types of faults
# NOTE: There are separate versions of the scalar and tensor values for portability
# If you add a new fault type, please create both the scalar and tensor functions 

import numpy as np

# Currently, we support 8 types of faults {None, Rand, Zero, Rand-element, bitFlip-element, bitFlip-tensor, binaryBitFlip, sequentialBitFlip} - See fiConfig.py

def randomScalar( dtype, max = 1.0 ):
	"Return a random value of type dtype from [0, max]"
	return dtype.type( np.random.random() * max )

def randomTensor( dtype, tensor):
	"Random replacement of a tensor value with another one"
	# The tensor.shape is a tuple, while rand needs linear arguments
	# So we need to unpack the tensor.shape tuples as arguments using *  
	res = np.random.rand( *tensor.shape ) 
	return dtype.type( res )

def zeroScalar(dtype, val):
	"Return a scalar 0 of type dtype"
	# val is a dummy parameter for compatibility with randomScalar
	return dtype.type( 0.0 )

def zeroTensor(dtype, tensor):
	"Take a tensor and zero it"
	res = np.zeros( tensor.shape ) 
	return dtype.type( res )

def noScalar(dtype, val):
	"Dummy injection function that does nothing"
	return val

def noTensor(dtype, tensor):
	"Dummy injection function that does nothing"
	return tensor

def randomElementScalar( dtype, max = 1.0):
	"Return a random value of type dtype from [0, max]"
	return dtype.type( np.random.random() * max )

def randomElementTensor ( dtype, val):
	"Random replacement of an element in a tensor with another one"
	"Only one element in a tensor will be changed while the other remains unchanged" 
	dim = val.ndim 
	
	if(dim==1):
		index = np.random.randint(low=0 , high=(val.shape[0]))
		val[index] = np.random.random() 
	elif(dim==2):
		index = [np.random.randint(low=0 , high=(val.shape[0])) , np.random.randint(low=0 , high=(val.shape[1]))]
		val[ index[0] ][ index[1] ] = np.random.random()

	return dtype.type( val )



def float2bin(number, decLength = 10): 
	"convert float data into binary expression"
	# we consider fixed-point data type, 32 bit: 1 sign bit, 21 integer and 10 mantissa

	# split integer and decimal part into seperate variables  
	integer, decimal = str(number).split(".") 
	# convert integer and decimal part into integer  
	integer = int(integer)  
	# Convert the integer part into binary form. 
	res = bin(integer)[2:] + "."		# strip the first binary label "0b"

	# 21 integer digit, 22 because of the decimal point "."
	res = res.zfill(22)
	
	def decimalConverter(decimal): 
		"E.g., it will return `x' as `0.x', for binary conversion"
		decimal = '0' + '.' + decimal 
		return float(decimal)

	# iterate times = length of binary decimal part
	for x in range(decLength): 
		# Multiply the decimal value by 2 and seperate the integer and decimal parts 
		# formating the digits so that it would not be expressed by scientific notation
		integer, decimal = format( (decimalConverter(decimal)) * 2, '.10f' ).split(".")    
		res += integer 

	return res 

def randomBitFlip(val):
	"Flip a random bit in the data to be injected" 

	# Split the integer part and decimal part in binary expression
	def getBinary(number):
		# integer data type
		if(isinstance(number, int)):
			integer = bin(int(number)).lstrip("0b") 
			# 21 digits for integer
			integer = integer.zfill(21)
			# integer has no mantissa
			dec = ''	
		# float point datatype 						
		else:
			binVal = float2bin(number)				
			# split data into integer and decimal part	
			integer, dec = binVal.split(".")	
		return integer, dec

	# we use a tag for the sign of negative val, and then consider all values as positive values
	# the sign bit will be tagged back when finishing bit flip
	negTag = 1
	if(str(val)[0]=="-"):
		negTag=-1

	if(isinstance(val, np.bool_)):	
		# boolean value
		return bool( (val+1)%2 )
	else:	
		# turn the val into positive val
		val = abs(val)
		integer, dec = getBinary(val)

	intLength = len(integer)
	decLength = len(dec)

	# random index of the bit to flip  
	index = np.random.randint(low=0 , high = intLength + decLength)
 
 	# flip the sign bit (optional)
	# if(index==-1):
	#	return val*negTag*(-1)

	# bit to flip at the integer part
	if(index < intLength):		
		# bit flipped from 1 to 0, thus minusing the corresponding value
		if(integer[index] == '1'):	val -= pow(2 , (intLength - index - 1))  
		# bit flipped from 0 to 1, thus adding the corresponding value
		else:						val += pow(2 , (intLength - index - 1))
	# bit to flip at the decimal part  
	else:						
		index = index - intLength 	  
		# bit flipped from 1 to 0, thus minusing the corresponding value
		if(dec[index] == '1'):	val -= 2 ** (-index-1)
		# bit flipped from 0 to 1, thus adding the corresponding value
		else:					val += 2 ** (-index-1) 

	return val*negTag

def bitElementScalar( dtype, val ):
	"Flip one bit of the scalar value"   
	return dtype.type( randomBitFlip(val) )

def bitElementTensor( dtype, val):
	"Flip ont bit of a random element in a tensor"
	# flatten the tensor into a vector and then restore the original shape in the end
	valShape = val.shape
	val = val.flatten()

	# select a random data item in the data space for injection
	index = np.random.randint(low=0, high=len(val))

	val[index] = randomBitFlip(val[index])	
	val = val.reshape(valShape)

	return dtype.type( val )

def bitScalar( dtype, val):
	"Flip one bit of the scalar value"
	return dtype.type( randomBitFlip(val) )

def bitTensor ( dtype, val):
	"Flip one bit of all elements in a tensor"
	# dimension of tensor value 
	dim = val.ndim		

	# the value is 1-dimension (i.e., vector)
	if(dim==1):			
		col = val.shape[0]
		for i in range(col):
			val[i] = randomBitFlip(val[i])

	# the value is 2-dimension (i.e., matrix)
	elif(dim==2):
		row = val.shape[0]
		col = val.shape[1]
		# flip one bit of each element in the tensor
		for i in range(row):
			for j in range(col): 
				val[i][j] = randomBitFlip(val[i][j]) 

	return dtype.type( val )


########################################## Binary FI
def initBinaryInjection(isFirstTime=True):
	"Initialize the values for binary fault injection"
	"NOTE: You should call this function before performing binary FI"

	# NOTE: You have to specify this value in the main program, based on the result from the FI
	# E.g., if the FI does not result in SDC, you should assign sdcFromLastFI=False, which will be used in the next FI run
	global sdcFromLastFI # SDC status from last FI, used for guiding next binary split
	global indexOfInjectedData # index of the data to be injected

	# we use binary search to decide to bit to be injected, requiring upper and lower indice for the bisecting the injection space
	global frontIndice # front indice for bisecting injection space
	global rearIndice # rear indice for bisecting injection space
	global indexOfBit_1	# list of the index of the bits of "1" of the current injected data
	global indexOfBit_0	# list of the index of the bits of "0" of the current injected data

	global isFIonBits0	# flag for whether still doing FI on the bits of 0  
	global isFIonBits1	# flag for whether still doing FI on the bits of 1
	global indexOfLastInjectedBit # index of the last injected bit

	global sdcRate # cumulative sdc rate at the current op, the sdc rate is calculated by using the SDC-boundary from bits of 0 and 1.
	global sdc_bound_1	# number of critical bits in the bits of 1 
	global sdc_bound_0	# number of critical bits in the bits of 0 
	
 	# indexOf_SDC_nonSDC_bit[0] points the indice of the critical bits, 
 	# indexOf_SDC_nonSDC_bit[1] points to that of the non-critical bits (if any)
 	# This is used to get the SDC bound
	global indexOf_SDC_nonSDC_bit 

	global isKeepDoingFI # sign for whether keep doing FI
	global fiTime 	# number of FI trials so far, i.e., the overhead of running binFI
	global isDoneForCurData # sign for whether the binary FI on the current data item is done
	global isFrom0to1 # sign for doing FI from bits of 0 to bits 1

 	# The first time to do FI on the current op
	if(isFirstTime): 
		indexOfInjectedData = 0	# start from the first data item in the output 
		sdcRate = 0.
		isKeepDoingFI = True
		fiTime = 0  
		isDoneForCurData = False 

	# initialization for the non-first time FI
	frontIndice = -1 # -1 is used as a sign for the initilization, normal value should be no smaller than 0
	rearIndice = -1
	indexOfBit_0 = [] # empty list of the index of the bits of 0
	indexOfBit_1 = []
	isFIonBits0 = False # default value, this will be updated when we're converting the data into binary expression
	isFIonBits1 = False # e.g., when the data consists of bits of 0, isFIonBits0 will be True while isFIonBits1 will be False
	indexOf_SDC_nonSDC_bit = [-1, -1] # -1 is used as a sign for the initilization, normal value should be no smaller than 0
	sdcFromLastFI = -1 # -1 is used as a sign for the initilization, normal value should be no smaller than 0
	indexOfLastInjectedBit = 0 
	isFrom0to1 = False # This only turns True when FI on the bits of 0 is done and is about to do FI on the bits of 1
	sdc_bound_1 = 0 # no critical bits, by default
	sdc_bound_0 = 0
	isDoneForCurData = False
	

def binFaultInjection(bitIndexToBeInjected, isBits0, injectedData, intLen):
	"Perform binary FI on the bits of 0 or 1"
	# bitIndexToBeInjected: is the list of indices of the bits, that will be injected.
	# 						The index is the actually index in the injected data, index from MSB to LSB. 
	#						E.g., 5 = 101B, so index of the bits of 1 will be [0, 2].
	
	global sdcFromLastFI # flag for whether the last FI cause SDC or not, used for choosing the next bit to be injected
	global indexOfLastInjectedBit # index of the last two injected bits 
	global frontIndice 
	global rearIndice  
	global indexOfInjectedBit # index of bit to be injected
	global fiTime 

	# this is the first FI entry
	if(frontIndice== -1 and rearIndice== -1):
		frontIndice = 0
		rearIndice = len(bitIndexToBeInjected)-1
	# Non-first FI entry: updating the front or rear indices based on the result from last FI
	elif(sdcFromLastFI==True):
		# last FI caused SDC, next FI moves to lower-order bits
		frontIndice =  bitIndexToBeInjected.index(indexOfLastInjectedBit) + 1
	elif(sdcFromLastFI==False):
		# last FI did not cause SDC, next FI moves to higher-order bits
		rearIndice = bitIndexToBeInjected.index(indexOfLastInjectedBit) - 1

	" terminate binary search: 1) front = rear; or 2) front > rear, which occurs when front and rear all pointed to the same indice"
	# indexOfLastInjectedBit could not be in bitIndexToBeInjected, when last FI is done for the bits of 0, and now the FI is moving to the bits of 1
	#  That's why we need (indexOfLastInjectedBit in bitIndexToBeInjected)
	if( ( (indexOfLastInjectedBit in bitIndexToBeInjected) and (frontIndice == rearIndice == bitIndexToBeInjected.index(indexOfLastInjectedBit)))
		or (frontIndice>rearIndice) ): 
 
		# binary FI at the bits of 0 is done, initialize the value for the FI on the bits of 1
		frontIndice = -1
		rearIndice = -1
		# return False means there is no FI in this call, will move FI to the bits of 1 at next call
		return False, 0
	else:
		# bisect and get the indece of the bit to be injected
		indexOfInjectedBit = bitIndexToBeInjected[ (frontIndice+rearIndice)/2 ]  
		fiTime += 1	# count the FI trial

		if(isBits0):
			# the data is negative and injected bit is '0', so the delta by bit flip is negative (-0 to -1)
			if(str(injectedData)[0] == "-"):
				afterBitFlip = injectedData - pow(2, intLen-(indexOfInjectedBit))
			# the data is positive and injected bit is '0', so the delta by bit flip is positive (+0 to +1)
			else:
				afterBitFlip = injectedData + pow(2, intLen-(indexOfInjectedBit))
		else:
			# the data is negative and injected bit is '1', so the delta by bit flip is positive (-1 to -0)
			if(str(injectedData)[0] == "-"):
				afterBitFlip = injectedData + pow(2, intLen-(indexOfInjectedBit))
			# the data is positive and injected bit is '1', so the delta by bit flip is negative (+1 to +0)
			else:
				afterBitFlip = injectedData - pow(2, intLen-(indexOfInjectedBit))  

	indexOfLastInjectedBit = indexOfInjectedBit # record the index of the current injected bit, used in next FI

	# return True means it'll keep doing FI for the current type of bits (0 or 1)
	# return the updated value after bit flip as well
	return True, afterBitFlip


def binaryBitFlip(dtype, val):  	
	"binary FI on both tensor and scalar values - we flatten the value (either tensor or scalar) and restore the original shape at the output"
	
	global indexOfInjectedData # index of the data to be injected 
	global indexOfBit_1 # list of the index of the bits of "1" of the current injected data
	global indexOfBit_0 # list of the index of the bits of "0" of the current injected data
	global isFIonBits0 # flag for whether still doing FI on the bits of 0  
	global isFIonBits1 # flag for whether still doing FI on the bits of 1
 	global sdcRate # cumulative SDC rate for the current op
 	global indexOfInjectedBit # index of the bit to be injected in the current data

 	# indexOf_SDC_nonSDC_bit[0] points the indice of the critical bits, [1] points to that of the non-critical bits (if any)
 	# This is used to derive the SDC bound
	global indexOf_SDC_nonSDC_bit 

	global indexOfLastInjectedBit # index of the last injected bit
	global isKeepDoingFI 
	global intLen # length of the integer bit 
	global sdcFromLastFI # FI result from last FI
	global isDoneForCurData # # sign for whether the binary FI on the current data item is done
	global isFrom0to1 # This only turns True when FI on the bits of 0 is done and is about to do FI on the bits of 1
	global sdc_bound_1	# number of bits that lead to SDC in the bits of 1
	global sdc_bound_0	# number of bits that lead to SDC in the bits of 1


	def updateSDC(bitIndexToBeInjected): 
		# derive the number of critical bits (i.e., the sdc boundary) based on binary FI

		global indexOf_SDC_nonSDC_bit  
		sdcBitCount = 0
		# There is no critical bits, as indexOf_SDC_nonSDC_bit[0] should store the indice of the latest bit that leads to SDC
		if(indexOf_SDC_nonSDC_bit[0] == -1):	sdcBitCount = 0
		# There are all critical bits, as [1] should store the indice of the latest bit that does not lead to SDC
		elif(indexOf_SDC_nonSDC_bit[1] == -1):	sdcBitCount = len(bitIndexToBeInjected) 
		else:
			try:
				# calculate the number of the critical bits 
				# e.g., bitIndexToBeInjected.index(indexOf_SDC_nonSDC_bit[0])=1 means the second bits in the data would cause SDC
				# this means that there are two critical bits in total
				sdcBitCount = bitIndexToBeInjected.index(indexOf_SDC_nonSDC_bit[0]) + 1
 			except:
 				# no SDC, if throw exception
 				sdcBitCount = 0

		return sdcBitCount


	# update the FI result from last FI (results or not results in SDC)
	# we Separate the case for the bits of 0 and 1, respectively, this is why we need the condition check of isFrom0to1
	if(sdcFromLastFI == True and (not isFrom0to1)):
		# update the index of the latest bit that causes SDC
		indexOf_SDC_nonSDC_bit[0] = indexOfLastInjectedBit 
	elif(sdcFromLastFI == False and (not isFrom0to1)):
		# update the index of the latest bit that does not cause SDC
		indexOf_SDC_nonSDC_bit[1] = indexOfLastInjectedBit


	# treat a scalar value as an array with one element, and thus the FI process is the same for scalar and tensor
	isScalar = np.isscalar(val)
	if(isScalar):
		val = np.atleast_1d(val)
	else:
		val = np.asarray(val, type(val))
		valShape = val.shape
		val = val.flatten() 


	injectedData = val[ indexOfInjectedData ] # data to be injected


	# create the index of the '1' and '0' bit of the current data
	if(indexOfBit_0==[] and indexOfBit_1==[]):

		if(isinstance(injectedData, int)):
			# "int" data type only has integer bit, but not manttissa
			binVal = bin(abs(injectedData)).lstrip("0b") 
			binVal = binVal.zfill(21)
			intLen = 20	# length of integer, this is the index, so the actual number is 20+1
		elif(isinstance(injectedData, float)):
			# "float" datatype has 21 integer-bits, 10 manttissa and 1 sign bit
			binVal = float2bin(abs(injectedData)).replace('.', '') 
			intLen = 20 # length of integer, this is the index, so the actual number is 20+1

		# collect the index of the bits of 0 and 1
		for index in range(len(binVal)):
			if(binVal[index] == '1'):
				indexOfBit_1.append(index)	# create the index of the bits of 0
				isFIonBits1 = True	# We will do FI on the bits of 1 if the data contains bits of 1
			elif(binVal[index] == '0'):
				indexOfBit_0.append(index) # create the index of the bits of 1
				isFIonBits0 = True # We will do FI on the bits of 0 if the data contains bits of 0


	"If a fault at higher bit does not cause SDC, faults at lower bits will not cause SDC - NOTE: see the SC19 paper for the rationale"
	"We separately do injection for the bits of 0 and 1 since they would have different impacts"
	# FI on the bits of 0
	if(isFIonBits0): 
		# perform binary FI
		isFIonBits0, updatedVal = binFaultInjection(bitIndexToBeInjected=indexOfBit_0, isBits0=True, injectedData=injectedData, intLen= intLen) 
		
		if(isFIonBits0):
			val[indexOfInjectedData] = updatedVal # return the value with fault injected
		else:
			sdc_bound_0 = updateSDC(indexOfBit_0) # sdcBoundary of the bits of 0 (number of critical bits in the bits of 0)
			indexOf_SDC_nonSDC_bit = [-1, -1] # intialize for FI on the bits of 1
			isFrom0to1 = True # FI is done in the bits of 0, and turning to bits of 1
	# FI on the bits of 1
	elif(isFIonBits1):
		isFrom0to1 = False # set to false for the current data, no longer needed. 

		# perform binary FI
		isFIonBits1, updatedVal = binFaultInjection(bitIndexToBeInjected=indexOfBit_1, isBits0=False, injectedData=injectedData, intLen= intLen) 

		if(isFIonBits1):
			val[indexOfInjectedData] = updatedVal # return the value with fault injection
		else:
			sdc_bound_1 = updateSDC(indexOfBit_1) # sdcBoundary of the bits of 1 (number of critical bits in the bits of 1)

	# FI for current data item is done. 
	if(isFIonBits0==False and isFIonBits1==False): 
 		isDoneForCurData = True # sign that FI on current data is done
		# cumulative SDC rate for the data at current Op
		# Noted that sdc_bound is the actually number of sdc bits, so the index of the actual sdc_bound should minus 1 (since the indice starts from 0)
		" (1) IF you want to calculate the SDC bound for the bits of 0 and 1, you should access the both variables in the main program, and then" 
		" call initBinaryInjection(isFirstTime=False) to do FI on the next data"
		" (2) IF you just want to know the SDC rate, you just need to access the sdcRate variable at the end of the FI in the main program"
		" In this case yhou don't need to init function in the main program, you can call the init function below"
		sdcRate += (sdc_bound_0 + sdc_bound_1) /  float((len(indexOfBit_1)+len(indexOfBit_0))*len(val))

 		# initialize the value for performing FI on the next datapoint (either here or in the main program)
		# initBinaryInjection(isFirstTime=False) 	# this is typically called in the main program since you've to record the values before initializing it
		indexOfInjectedData += 1 # index of next data to be injected


	if(indexOfInjectedData < len(val)):
		# This is not the last data item in the current op
		isKeepDoingFI = True
	else:
		# This is the last data item in the curernt op, so the FI is done for current op
		isKeepDoingFI = False
 
 	# return the output of the op with fault injected
	if(not isScalar):
		return dtype.type( val.reshape(valShape) )
	else:
		return dtype.type(val[0])


################################## Exhaustive FI
def sequentialFIinit():
	"Initialize the values for exhaustive fault injection"
	"You should call this function before performing exhaustive FI"
	global indexOfInjectedData # index of the data to be injected, starting from the first data
	global indexOfInjectedBit # index of the bit to be injected, starting from the first bit
	global isKeepDoingFI # sign for whether keep doing FI
	isKeepDoingFI = True
	indexOfInjectedData = 0
	indexOfInjectedBit = 0
 

def sequentialBitFlip(dtype, val): 
	global indexOfInjectedData 
	global indexOfInjectedBit 
	global isKeepDoingFI
 
 	# treat a scalar value as an array with one element, and thus the FI process is the same for scalar and tensor
	isScalar = np.isscalar(val)
	if(isScalar):
		val = np.atleast_1d(val)
	else:
		val = np.asarray(val, type(val))
		valShape = val.shape
		val = val.flatten() 

	injectedData = val[ indexOfInjectedData ] # data to be injected

	if(isinstance(injectedData, int)):
		# "int" data type only has integer bit, but not manttissa
		binVal = bin(abs(injectedData)).lstrip("0b") 
		binVal = binVal.zfill(21)
		maxIndex = 20
	elif(isinstance(injectedData, float)):
		# "float" datatype has 21 integer-bits, 10 manttissa and 1 sign bit
		binVal = float2bin(abs(injectedData)).replace('.', '')  
		maxIndex = 30


	if(str(injectedData)[0] == "-"):
		if(binVal[indexOfInjectedBit] == '0'):
			# flip 0 to 1 in a negative data, the delta is negative
			injectedData -= pow(2, (20- indexOfInjectedBit))
		else:
			# flip 1 to 0 in a negative data, the delta is is positive
			injectedData += pow(2, (20- indexOfInjectedBit))
	else:
		if(binVal[indexOfInjectedBit] == '0'):
			# flip 0 to 1 in a positive data, the delta is positive
			injectedData += pow(2, (20- indexOfInjectedBit))
		else:
			# flip 1 to 0 in a positive data, the delta is negative
			injectedData -= pow(2, (20- indexOfInjectedBit))

	# update the injected value
	val[indexOfInjectedData] = injectedData 

	indexOfInjectedBit += 1 # index of the next bit to be injected
	
	if(indexOfInjectedBit > maxIndex):
		# this is last bit to be injected in the current data, move FI to the next data item
		indexOfInjectedData += 1
		indexOfInjectedBit = 0

	# Note: If you want to constraint your exhaustive FI on a certain number of values, you can change the following variable
	numOfValueToInject = len(val)	
	if(indexOfInjectedData < numOfValueToInject):
		# This is not the last data item in the current op
		isKeepDoingFI = True
	else:
		# This is the last data item in the curernt op, so the FI is done for current op
		isKeepDoingFI = False
 

	if(not isScalar):
		return dtype.type( val.reshape(valShape) )
	else:
		return dtype.type(val[0])






