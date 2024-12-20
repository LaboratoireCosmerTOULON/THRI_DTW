import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
import yaml
import os
import DataLoader as lib
import pandas as pd

def save_dfs(data_df,data_loader,diver,prefix,data_loc,deriv = False):
    one_diver_Loader = lib.DataLoader(direc='./',gesture_array=[''])
    df_monogestures = data_df.loc[data_df['twohands'] == 0]
    df_duogestures = data_df.loc[data_df['twohands'] == 1]
    print("For ", diver,prefix," data we have ",len(data_df), "unique signals in total : \n",
          'One armed signals: ', len(df_monogestures), 
          '\nTwo armed signals: ', len(df_duogestures))
    if (len(data_df) != (len(df_monogestures) + len(df_duogestures))): 
        print('diver ', diver,' data is incorrect' ) 
    #df_monogestures = df_monogestures[ df_monogestures['original'] =='original']
    for  i in range (len(data_loader.right_side_feature)): 
        df_monogestures.loc[df_monogestures['h']==1,data_loader.right_side_feature[i]] = df_monogestures[data_loader.left_side_feature[i]]
    
    df_monogestures =df_monogestures.drop(data_loader.left_side_feature,axis=1)
    one_diver_Loader.df = df_monogestures
    one_diver_Loader._add_meta_columns(['twohands','h'])
    if deriv==True: 
        one_diver_Loader.calculate_gesture_velocity(alpha=0.3,beta=0.1,keep_signal= False)
    #one_diver_Loader.standarize_df()
    df_monogestures = one_diver_Loader.df
    one_diver_Loader.df = df_duogestures
    if deriv==True: 
        one_diver_Loader.calculate_gesture_velocity(alpha=0.3,beta=0.1,keep_signal= False)
    #one_diver_Loader.standarize_df()
    df_duogestures = one_diver_Loader.df
    df_monogestures = df_monogestures.drop(columns=['twohands','h']) 
    df_duogestures = df_duogestures.drop(columns=['twohands','h']) 
    df_monogestures.to_pickle(data_loc+'mono/'+diver+'_'+prefix+'.pickle')
    df_duogestures.to_pickle(data_loc+'duo/'+diver+'_'+prefix+'.pickle')

def  svm_classifier(data_df,_energy_df,name,output_direc,data_loader): 
    train_df = data_df[data_df['Name']!=name]
    train_energy_df = _energy_df[_energy_df['Name']!=name]
    test_df= data_df[(data_df['Name']==name) & (data_df['original']=='original')]
    test_energy_df = _energy_df[(_energy_df['Name']==name) &(_energy_df['original'] == 'original')]
    train_df.to_pickle(output_direc+'nosep/'+name+'_train.pickle') 
    test_df.to_pickle(output_direc+'nosep/'+name+'_test.pickle') 
    mapping_onetwo = {'L': 0, 'R': 0, 'B': 1}
    mapping_h ={'L':1,'R':2,'B':2}
    clf = SVC(C=100, kernel='poly',degree=2,  random_state=5)
    train_energy_df['y'] = train_energy_df['hands'].map(mapping_onetwo)
    train_df['twohands'] = train_df['hands'].map(mapping_onetwo)
    train_df['h'] = train_df['hands'].map(mapping_h)
    train_energy_df =train_energy_df.dropna()
    x1 = train_energy_df['x1'].values
    x2 = train_energy_df['x2'].values
    X = np.stack([x1,x2]).T
    y = train_energy_df['y'].values
    clf.fit(X, y)
    x1 = test_energy_df['x1'].values
    x2 = test_energy_df['x2'].values
    X = np.stack([x1,x2]).T
    test_df['twohands'] = clf.predict(X)
    test_energy_df['twohands'] = clf.predict(X)
    test_df['h'] = test_energy_df['h'].values
    save_dfs(test_df,data_loader, name,prefix='test' ,data_loc=output_direc)
    save_dfs(train_df,data_loader, name,prefix='train' ,data_loc=output_direc)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml',required=True, dest ='yaml_file', type=str)
    args =parser.parse_args() 
    with open (args.yaml_file, 'r') as yaml_file:
        config_data = yaml.load(yaml_file,Loader= yaml.FullLoader)
        divers = config_data['divers']
        label  = config_data['gestures'] 

    input_direc = config_data['dataloc']
    output_direc = config_data['outputloc']
    data_loader = lib.DataLoader(direc=input_direc,gesture_array=label, divers=divers)
    
    data_loader.load_data_with_extra_features(direc= input_direc)
    data_loader.initialize_from_yaml(args.yaml_file)
    data_loader.df['meta'] = data_loader.df['signal_id']+data_loader.df['original']
    duplicates = data_loader.df[data_loader.df['meta'].duplicated(keep=False)]
    
    data_loader.df =data_loader.df.drop(columns=['meta'])
    for i in range(0,len(divers)): 
        diver_df = data_loader.df[data_loader.df['Name']==divers[i]]
        print('for ', divers[i], 'we have: ',len(diver_df),'gesture including data augmentation')
    _energy_df = data_loader.get_energy_right_left_df()
    
    data_df = data_loader.df
    directories = ["mono", "duo", "nosep"]

# Create each directory if it doesn't exist
    for directory in directories:
        if not os.path.exists(output_direc+directory):
            os.mkdir(output_direc+directory)
    
    for name  in (divers):
        print('processing  ',name)
        svm_classifier(data_df=data_df,_energy_df=_energy_df,name= name,output_direc=output_direc,data_loader = data_loader)
        # print(name)

if __name__ == "__main__": 
    main()         