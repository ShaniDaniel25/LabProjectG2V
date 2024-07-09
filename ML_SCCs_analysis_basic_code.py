# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:14:05 2024

@author: shani
"""

import sys
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


N_JOBS = 10
ABS_PATH = "/davidb/shanidaniel1/" # from phoenix server

# creating instances of classifiers (\models)
GradientBoostingClassifier_model = GradientBoostingClassifier(random_state=42)
RandomForestClassifier_model = RandomForestClassifier(random_state=42, n_jobs = N_JOBS, class_weight = "balanced")
SVC_model = SVC(probability=True, random_state=42, class_weight = "balanced") # need probability = True for roc curve
KNeighborsClassifier_model = KNeighborsClassifier(n_jobs = N_JOBS)

base_estimator = SVC(kernel='linear', probability=True, class_weight = "balanced")
adaboost_classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)


def main():
    # Check if at least four command-line arguments are provided
    if len(sys.argv) < 4:
        print("Usage: python your_script.py arg1 arg2 arg3")
        sys.exit(1)

    # Extract the first three arguments
    X_matrix_csv = str(sys.argv[1])
    y_vector_csv = str(sys.argv[2])
    KO_groups_csv = str(sys.argv[3])
    
    # Load data from CSV files using pandas
    X_matrix = pd.read_csv(X_matrix_csv)
    X_matrix = X_matrix.astype(float)
    y_vector = pd.read_csv(y_vector_csv)
    KO_groups = pd.read_csv(KO_groups_csv)

    n = int(y_vector.nunique()) # number of SCCs 
    ticks_cf = [i+0.5 for i in range(n)]
    labels = sorted(y_vector['single_component'].unique())

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) # 5 non-overlapping tests, meaning around 20% test - 80% train each
    sgkf_split = sgkf.split(X_matrix, y_vector, KO_groups) # splits while stratifying y and keeping same KO's together
    
    for i, (train_index, test_index) in enumerate(sgkf_split):
        if i==0: # first fold
    
            X_train, X_test = X_matrix.loc[train_index], X_matrix.loc[test_index]
            y_train, y_test = y_vector.loc[train_index], y_vector.loc[test_index]
    
            print("SVC")
            cf_matrix = test_and_evaluate_classifier(SVC_model, X_train, X_test, y_train, y_test, labels)
            print(cf_matrix)
            print("plotting")
            # Adjust x and y labels to start from 1 instead of 0
            plt.figure(figsize=(20, 16))
            sns.heatmap(cf_matrix, annot=False)
            # Modify x and y tick labels to count from 1 to n+1
            plt.xticks(ticks=ticks_cf, labels=labels)
            plt.yticks(ticks=ticks_cf, labels=labels)
            plt.title("Confusion Matrix - SVC")
            plt.savefig(f'{ABS_PATH}SVC_heatmap.png')
            print("computing roc curves")
            compute_roc_curve(SVC_model, X_train, X_test, y_train, y_test)
            

            print("K Nearest Neighbors Classifier")
            cf_matrix = test_and_evaluate_classifier(KNeighborsClassifier_model, X_train, X_test, y_train, y_test, labels)
            print(cf_matrix)
            print("plotting")
            plt.figure(figsize=(20, 16))
            sns.heatmap(cf_matrix, annot=False)
            # Modify x and y tick labels to count from 1 to n+1
            plt.xticks(ticks=ticks_cf, labels=labels)
            plt.yticks(ticks=ticks_cf, labels=labels)
            plt.title("Confusion Matrix - K Nearest Neighbors")
            plt.savefig(f'{ABS_PATH}K_Nearest_Neighbors_heatmap.png')
            print("computing roc curves")
            compute_roc_curve(KNeighborsClassifier_model, X_train, X_test, y_train, y_test)


            print("Random Forest Classifier")
            cf_matrix = test_and_evaluate_classifier(RandomForestClassifier_model, X_train, X_test, y_train, y_test, labels)
            print(cf_matrix)
            print("plotting")
            plt.figure(figsize=(20, 16))
            sns.heatmap(cf_matrix, annot=False)
            # Modify x and y tick labels to count from 1 to n+1
            plt.xticks(ticks=ticks_cf, labels=labels)
            plt.yticks(ticks=ticks_cf, labels=labels)
            plt.title("Confusion Matrix - Random Forest")
            plt.savefig(f'{ABS_PATH}Random_Forest_heatmap.png')
            print("computing roc curves")
            compute_roc_curve(RandomForestClassifier_model, X_train, X_test, y_train, y_test)
            
     
            print("Gradiant Boosting")
            cf_matrix = test_and_evaluate_classifier(GradientBoostingClassifier_model, X_train, X_test, y_train, y_test, labels)
            print(cf_matrix)
            print("plotting")
            plt.figure(figsize=(20, 16))
            sns.heatmap(cf_matrix, annot=False)
            # Modify x and y tick labels to count from 1 to n+1
            plt.xticks(ticks=ticks_cf, labels=labels)
            plt.yticks(ticks=ticks_cf, labels=labels)
            plt.title("Confusion Matrix - Gradiant Boosting")
            plt.savefig(f'{ABS_PATH}Gradiant_Boosting_heatmap.png')
            print("computing roc curves")
            compute_roc_curve(GradientBoostingClassifier_model, X_train, X_test, y_train, y_test)
            
            
            print("Ada Boost Classifier")
            cf_matrix = test_and_evaluate_classifier(adaboost_classifier, X_train, X_test, y_train, y_test, labels)
            print(cf_matrix)
            print("plotting")
            plt.figure(figsize=(20, 16))
            sns.heatmap(cf_matrix, annot=False)
            # Modify x and y tick labels to count from 1 to n+1
            plt.xticks(ticks=ticks_cf, labels=labels)
            plt.yticks(ticks=ticks_cf, labels=labels)
            plt.title("Confusion Matrix - Ada Boost")
            plt.savefig(f'{ABS_PATH}Ada_Boost_heatmap.png')
            print("computing roc curves")
            compute_roc_curve(adaboost_classifier, X_train, X_test, y_train, y_test)
            

def test_and_evaluate_classifier(classifier, X_train, X_test, y_train, y_test, labels):
    print(f'begin fitting {classifier}')
    classifier.fit(X_train, y_train)
    print(f'ended fitting {classifier}')
    y_pred = classifier.predict(X_test)
    print('created y_pred')
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Display results
    print(f"Classifier: {type(classifier).__name__}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
    print("="*50)
    
    print("computing confusion matrix")
    cf_matrix = confusion_matrix(y_test, y_pred, labels = labels)
    print("computed confusion matrix")
    
    return cf_matrix


def compute_roc_curve(classifier, X_train, X_test, y_train, y_test):
    # Predict probabilities for each class
    y_proba = classifier.predict_proba(X_test)
    print("creating 1vr classifier")
    ovr_classifier = OneVsRestClassifier(classifier)
    print("beginning fit")
    ovr_classifier.fit(X_train, y_train) 
    print("finished fitting")
    # Binarize the labels for ROC curve (assuming binary or multilabel classification)
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, class_label in enumerate(np.unique(y_test)):
        y_test_bin_i = y_test_bin[:, i]
        fpr[class_label], tpr[class_label], _ = roc_curve(y_test_bin_i, ovr_classifier.predict_proba(X_test)[:, i])
        roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
    
    plt.figure(figsize=(10, 10))
    for i, class_label in enumerate(np.unique(y_test)):
        plt.plot(fpr[class_label], tpr[class_label], label='ROC curve (class %s, AUC = %0.2f)' % (class_label, roc_auc[class_label]))
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {classifier} (One-vs-Rest)')
    # plt.legend(loc="lower right")
    plt.savefig(f'{ABS_PATH}ROC_{str(classifier)[:30]}.png')  # Save the plot to a file

    # AUPR curve
    
    # Calculate Precision-Recall curve for each class
    precision = dict()
    recall = dict()
    thresholds = dict()
    for i in range(len(classifier.classes_)):
        precision[i], recall[i], thresholds[i] = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
    
    # Plot Precision-Recall curve for each class
    plt.figure(figsize=(16, 12))
    for i in range(len(classifier.classes_)):
        plt.plot(recall[i], precision[i], label='Class {}'.format(classifier.classes_[i]))
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {classifier} (One-vs-Rest)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(f'{ABS_PATH}AUPR_{str(classifier).split("(")[0]}.png')  # Save the plot to a file


if __name__ == "__main__":
    main()