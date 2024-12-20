import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from KnnDtw import KnnDtw
import pandas as pd
from sklearn.svm import SVC
import pickle
import csv
import DataLoader as lib
from KnnDtw import KnnDtw
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def main(): 
    '''Arguments explanation: 
    --train_set    >> directory for 
    --df           >> df containing signals
    --division_type>> what division type are we using  '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--window',required = False, dest = 'window',type=int, default=10,  help = 'length of  sliding window' )
    parser.add_argument('--neighbors',required = False, dest = 'neighbors',type=int, default=3,  help = 'number of neighbors used for comparaison' )
    parser.add_argument('--distance_type',required = False, dest = 'distance',type=str, default='angle',  help = 'type of error calcuclation' )
    parser.add_argument('--subject',dest='subject', type=str, required=True )
    parser.add_argument('--data_directory', dest = 'data_directory',type=str , required =False, default = 'nosep/')
    parser.add_argument('--yaml',dest='yaml', type=str, required=False )
    parser.add_argument('--output',dest='output', type=str, required=False )
    parser.add_argument('--signal_ith', dest='ith' ,type= int, required  =True, default = 0 )
    args =parser.parse_args() 
    window = args.window 
    neighbors = args.neighbors
    distance_type = args.distance
    distance_type = args.distance
    print(args.data_directory)
    print('widows length KNN: ', window)
    print('distance type used: ',distance_type)
    Loader = lib.DataLoader(direc='./',meta_columns=['signal_id','oscillations','position','rest_position','hands','conform','original','Name'] )
    Loader.initialize_from_yaml(args.yaml)
    m = KnnDtw(n_neighbors=neighbors, max_warping_window=window,subsample_step=1,class_names=Loader.gesture_array, distance_type=distance_type ,wrapping_calculation=True,output = args.output)
    subject = args.subject
    Loader.load_df(Loader.direc+args.data_directory+subject+'_test.pickle')
    meta_test,test  = Loader.seperate_additional_data()
    signal_id = test['signal_id'].iloc[args.ith]
    test = test.drop(columns=Loader.meta_columns)
    print('the subject to test:', test)
    Loader.load_df(Loader.direc+args.data_directory+subject+'_train.pickle')
    meta_train,train  = Loader.seperate_additional_data() 
    train = train.drop(columns=Loader.meta_columns)
    
    X_test, Y_test,Y_test_class, Y_test_oh  = Loader.split_x_y(test,class_number =len(Loader.gesture_array))
    X_test = Loader.transform_X_data(X_test)
    test = X_test[args.ith]
    meta_test = meta_test.iloc[args.ith]
    X_train, Y_train,Y_train_class, Y_train_oh  = Loader.split_x_y(train,class_number =len(Loader.gesture_array))
    X_train = Loader.transform_X_data(X_train)
    #X_val = Loader.transform_X_data(X_val)
    m.fit(X_train,Y_train_class,meta_train,Y_train)
    _=  m.predict_verbose_sep(test,meta_test, suffix = signal_id)



if __name__ == "__main__":
    main()
