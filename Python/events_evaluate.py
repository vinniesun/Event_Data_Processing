from cmath import nan
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import scikitplot as skplt
import math

EFAST_FILE = "eFast_Result.csv"
ARCSTAR_FILE = "ArcStar_Result.csv"

GROUND_TRUTH = "/Users/vincent/Desktop/CityUHK/EBBINNOT/Output/Shapes_Rotation_Absolute/"
LOCATION = "/Users/vincent/Desktop/CityUHK/EBBINNOT/Output/"
PROCESS = ["Shapes_Rotation_Bits_12/", "Shapes_Rotation_Bits_16/", "Shapes_Rotation_Bits_18/", "Shapes_Rotation_Bits_20/",
            "Shapes_Rotation_Bits_24/", "Shapes_Rotation_Delta_33000/", "Shapes_Rotation_Delta_66000/"]

TOTAL_EVENTS = 100000

#CLASS = {"TP" : 0, "TN" : 1, "FP" : 2, "FN" : 3}
CLASS = {"NOT CORNER" : 0, "CORNER" : 1}

def process_csv(filename):
    with open(filename, 'r') as f:
        num_events = 0
        for _ in f:
            num_events += 1

    result = [None] * (num_events - 1) # To skip the header row
    with open(filename, 'r') as f:
        for i, line in tqdm(enumerate(f)):
            if i != 0:
                event = line.split(",")
                result[i-1] = (event[0], event[1], event[2], event[3].strip("\n"))

    return num_events-1, result

def perf_metrics(y_actual, y_hat,threshold):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(len(y_hat)): 
        if(y_hat[i] >= threshold):
            if(y_actual[i] == 1):
                tp += 1
            else:
                fp += 1
        elif(y_hat[i] < threshold):
            if(y_actual[i] == 0):
                tn += 1
            else:
                fn += 1
    
    #We find the True positive rate and False positive rate based on the threshold
            
    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)

    return [fpr,tpr]

def main(filetype):
    print("Start Evaluating")
    print("---------------------------------------------------------------")

    """
        Process eFast First
    """
    print(filetype)
    # Get Ground Truth
    ground_truth_corners_count, ground_truth_corners = process_csv(GROUND_TRUTH + filetype)
    print("There are " + str(ground_truth_corners_count) + " Ground Truth Corners, out of " + str(TOTAL_EVENTS))
    print("---------------------------------------------------------------")
    
    for p in PROCESS:
        current_corner_count, current_corner = process_csv(LOCATION + p + filetype)
        
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        #y_pred, y_actual = [CLASS["TN"]] * TOTAL_EVENTS, [CLASS["TN"]] * TOTAL_EVENTS
        y_pred, y_actual = [CLASS["NOT CORNER"]] * TOTAL_EVENTS, [CLASS["NOT CORNER"]] * TOTAL_EVENTS
        i = 0
        for c in current_corner:
            if c in ground_truth_corners:
                true_pos += 1
                #y_pred[i] = CLASS["TP"]
                #y_actual[i] = CLASS["TP"]
                y_pred[i] = CLASS["CORNER"]
                y_actual[i] = CLASS["CORNER"]
            else:
                false_pos += 1
                #y_pred[i] = CLASS["FP"]
                #y_actual[i] = CLASS["TP"]
                y_pred[i] = CLASS["NOT CORNER"]
                y_actual[i] = CLASS["CORNER"]
            i += 1

        false_neg = ground_truth_corners_count - true_pos - false_pos

        #for j in range(i, i+false_neg):
        #    y_pred[j], y_actual[j] = CLASS["FN"], CLASS["TP"]

        true_neg = TOTAL_EVENTS - true_pos - false_pos - false_neg
        print("For " + p + ":")
        print("True Positive = " + str(true_pos) + ", True Negative = " + str(true_neg) + ", False Positive = " + str(false_pos) + ", False Negative = " + str(false_neg))

        tpr, specificity, fpr = true_pos/(true_pos + false_neg), true_neg/(true_neg + false_pos), false_pos/(true_neg + false_pos)
        print("Calculated True Positive Rate = " + str(tpr) + ", Specificity = " + str(specificity) + ", False Positive Rate = " + str(fpr))

        #fpr_package, tpr_package, threshold = metrics.roc_curve(y_actual, y_pred)
        #print("Sklearn True Positive Rate = " + str(tpr_package) + ", False Positive Rate = " + str(fpr_package))
        #roc_auc_calc = metrics.auc(fpr, tpr)
        #roc_auc_package = metrics.auc(fpr_package, tpr_package)
        #print("ROC_AUC_CALC = " + str(0) + ", ROC_AUC_PACKAGE = " + str(roc_auc_package))
        #print("ROC_AUC_CALC = " + str(roc_auc_package))

        thresholds = [0,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,1]

        roc_points = []
        for threshold in thresholds:
            rates = perf_metrics(y_actual, y_pred, threshold)
            roc_points.append(rates)

        fpr_array = []
        tpr_array = []
        for i in range(len(roc_points)-1):
            point1 = roc_points[i]
            point2 = roc_points[i+1]
            tpr_array.append([point1[0], point2[0]])
            fpr_array.append([point1[1], point2[1]])

        auc = sum(np.trapz(tpr_array,fpr_array))+1 # use Trapezoidal rule to calculate the area under the curve and approximating the intergral

        fig = plt.figure(figsize=(16,10))
        plt.plot(tpr_array,fpr_array, 'r', lw=2)
        plt.savefig("../Output/AUC_ROC of " + p + ".jpg", dpi=300)
        plt.show()
        plt.close()

        #print(roc_auc_package, type(roc_auc_package))
        #if math.isnan(roc_auc_package):
        #    continue

        #fig = plt.figure(figsize=(16,10))
        #skplt.metrics.plot_roc_curve(y_actual, y_pred)
        #plt.savefig("../Output/AUC_ROC of " + p + ".jpg", dpi=300)
        #plt.show()
        #plt.close()
        print("---------------------------------------------------------------")

if __name__ == "__main__":
    main(EFAST_FILE)
    main(ARCSTAR_FILE)