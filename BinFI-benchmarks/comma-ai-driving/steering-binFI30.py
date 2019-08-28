""" 
comma.ai steering model, model structure is in model.py

Model implementation based on https://github.com/commaai/research/blob/master/train_steering_model.py

We modify the Dave model in https://github.com/SullyChen/Autopilot-TensorFlow to accommodate the comma.ai model

Dataset available from https://github.com/SullyChen/driving-datasets 
"""


import tensorflow as tf
import scipy.misc
import model
#import cv2
from subprocess import call
import driving_data
import time
import TensorFI as ti
import datetime
sess = tf.InteractiveSession()
saver = tf.train.Saver() 
saver.restore(sess, "save/model.ckpt")  # restore the trained model

#img = scipy.misc.imread('steering_wheel_image.jpg',0)
#rows,cols = img.shape
#smoothed_angle = 0

fi = ti.TensorFI(sess, logLevel = 50, name = "convolutional", disableInjections=True)

# threshold deviation to define SDC
sdcThreshold = 30

# save FI results into file, "eachRes" saves each FI result, "resFile" saves SDC rate
resFile = open(`sdcThreshold` + "autopilot-binFI.csv", "a") 
eachRes = open(`sdcThreshold` + "each-binFI.csv", "a")

# inputs to be injected
index = [20, 486, 992, 1398, 4429, 5259, 5868, 6350, 6650, 7771]
#while(cv2.waitKey(10) != ord('q')):
  
for i in index:
    full_image = scipy.misc.imread("driving_dataset/" + str(i) + ".jpg", mode="RGB")
    image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0

    '''    
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi 
#    call("clear")
    print(i , ".png", " Predicted steering angle: " + str(degrees) + " degrees", driving_data.ys[i])
    resFile.write(`i` + "," + `degrees` + "," + `driving_data.ys[i]` + "\n")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1 
    '''

    # we first need to obtain the steering angle in the fault-free run
    fi.turnOffInjections()
    degrees = model.y.eval(feed_dict={model.x: [image]})[0][0] * 180.0 / scipy.pi 
    golden = degrees
    print(i , ".png", " Predicted steering angle: " + str(degrees) + " degrees", driving_data.ys[i])

    # perform FI
    fi.turnOnInjections()
    # initiliaze for binary FI
    ti.faultTypes.initBinaryInjection()
 
    totalFI = 0.  
    while(ti.faultTypes.isKeepDoingFI):
        degrees = model.y.eval(feed_dict={model.x: [image]})[0][0] * 180.0 / scipy.pi 

        totalFI += 1 
        # you need to feedback the FI result to guide the next FI for binary search
        if(abs(degrees - golden) < sdcThreshold):   
            # the deviation is smaller than the given threshold, which does not constitute an SDC 
            ti.faultTypes.sdcFromLastFI = False
        else:    
            ti.faultTypes.sdcFromLastFI = True  

        # if FI on the current data item, you might want to log the sdc bound for the bits of 0 or 1
        # (optional), if you just want to measure the SDC rate, you can access the variable of "ti.faultTypes.sdcRate"
        if(ti.faultTypes.isDoneForCurData):
            eachRes.write(`ti.faultTypes.sdc_bound_0` + "," + `ti.faultTypes.sdc_bound_1` + ",")
            # Important: initialize the binary FI for next data item.
            ti.faultTypes.initBinaryInjection(isFirstTime=False)
            
        print(i, totalFI)

    resFile.write(`ti.faultTypes.sdcRate` + "," + `ti.faultTypes.fiTime` + "\n")
    print(ti.faultTypes.sdcRate , "fi time: ", ti.faultTypes.fiTime) 

    
#        cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
#        #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
#        #and the predicted angle
#        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
#        M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
#        dst = cv2.warpAffine(img,M,(cols,rows))
#        cv2.imshow("steering wheel", dst)
        
    
#cv2.destroyAllWindows()
