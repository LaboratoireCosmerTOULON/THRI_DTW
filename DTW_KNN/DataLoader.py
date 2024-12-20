from base64 import standard_b64decode
from calendar import c
import numpy as np 
import pandas as pd 
import pickle
from scipy import signal as sig
import yaml
import matplotlib.pyplot as plt
import itertools
from pathlib import Path 
from sklearn.metrics import accuracy_score, confusion_matrix
from utils.AlphaBeta import AlphaBetaFilter,Sample, FrequencyFilter
import copy
import random
from utils.sigmoid import Sigmoid
class DataLoader: 
    def __init__(self,direc = './', divers = ['subject1','subject2','subject3'], \
        gesture_array=  ['godown','reserve','goup'],\
            meta_columns = ['label', 'signal_id', 'oscillations', 'position','rest_position', 'hands', 'conform', 'original', 'Name']): 
        self.gesture_array= gesture_array 
        self.class_names  =  {gesture_array[i]:i for i in range(len(gesture_array))}
        self.divers = divers
        self.direc = direc
        self.meta_columns = meta_columns
        self.extra_columns = []
        self.df = pd.DataFrame()
        
    ## methods used to fill in our data
    def initialize_from_yaml(self,file_path='./.yaml'): 
        '''initilazation code for paramaters that can be stored in a yaml file'''
        print(file_path)
        with open(file_path, 'r') as yaml_file:
            config_data = yaml.load(yaml_file, Loader=yaml.FullLoader)   
        self.gesture_array = config_data['gestures'] 
        self.class_names={element:index for index,element in enumerate(self.gesture_array)}
        self.subjects = config_data['divers']
        self.subject_anon_identifier = {subject:index  for index,subject in enumerate(self.subjects)}
        self.direc=config_data['dataloc']
        self.right_side_feature=list(map(str, config_data['right_side_features']))
        self.left_side_feature=list(map(str,config_data['left_side_features']))
        self.all_features = self.right_side_feature+self.left_side_feature
        print(self.all_features)

    def _add_meta_columns(self,column_arr): 
        for column  in column_arr: 
            self.meta_columns.append(column)
    def _get_dataframe(self): 
        return self.df
    def _get_class_names(self):
        return self.gesture_array
    def augment_diver_data(self,diver_data,diver): 
        '''loads df data, add unique identifier for each signal (among signal of particular diver) 
        explictly add the diver name'''
        #Is this still needed ? 
        diver = diver.lower()
        print(diver)
        if self.df.empty == False: 
            if (diver in set(self.df['Name']) ) : 
                ashta = self.df.loc[(self.df['Name'] == diver)]
                max_x = ashta.loc[ashta['signal_id'].idxmax()]
                counter = (max_x['signal_id'])+1
                diver_data.insert(0,'signal_id', range(counter,counter+len(diver_data)))
            else:
                diver_data.insert(0, 'signal_id', range(0,0+ len(diver_data)))
        else:
            diver_data.insert(0, 'signal_id', range(0,0+ len(diver_data)))
        diver_data = diver_data.assign(Name = diver)
        return diver_data

    def load_df(self,file): 
        with open (file, 'rb') as f: 
            df = pickle.load(f)
        self.df = df

    def load_data_with_extra_features(self,suffix='.pickle',direc='./'): 
        # IS it still needed? yes still needed since it filters out according to the gestures
        # this is also good for seperating the content on multiple campaign
        '''loads all the data in the given directory with the extra data to make each signal unique'''
        data =[]
        for  diver in self.divers:
            file = direc+diver+suffix
            print(file)
            path = Path (file)
            if path.is_file():
                with open (file,'rb') as f:
                    diver_data = pickle.load(f)
                diver_data = diver_data[diver_data['label'].isin(self.gesture_array)]
                diver_data = diver_data[diver_data['conform']!='N']
                #diver_data = self.augment_diver_data(diver_data,diver)
                data.append(diver_data)
        df = pd.concat(data,ignore_index =True)
        if self.df.empty:
            self.df = df
        else: 
            self.df = pd.concat([self.df, df],ignore_index=True) 
    
    def extract_diver(self,diver_name): 
        '''extract only one diver'''
        df = self.df
        diver_df = df.loc[df['Name'] == diver_name]
        return diver_df,df['signal_id'].tolist()

    def extract_row_using_panda_idx(self,one_signal,with_extra = True): 
        label = one_signal['label']
        if with_extra == True : 
            one_signal = one_signal.drop(columns=self.meta_columns).values
        if with_extra == False : 
            one_signal = one_signal[self.all_features].values
        one_signal = [np.vstack(item)for item in (one_signal)]
        one_signal = np.array(one_signal)
        return one_signal
    
    def extract_row(self, diver_name, signal_id,with_extra = True ,with_label = False, copy='original'): 
        'extract only one signal from our df'
        df = self.df
        one_signal = df.loc[(df['signal_id'] == signal_id) & (df['Name'] == diver_name )& (df ['original'] ==copy)]
        
        # print(one_signal['label'])
        label = one_signal['label']
        if with_extra == True : 
            one_signal = one_signal.drop(columns=self.meta_columns).values
        if with_extra == False : 
            one_signal = one_signal[self.all_features].values
        one_signal = [np.vstack(item)for item in (one_signal)]
        one_signal = np.array(one_signal)
        if with_label: 
            return one_signal,diver_name,signal_id,label
        else:
            return one_signal,diver_name,signal_id
    
    
    def transform_X_data(self,train_data): 
        '''transfrom X data from the df shape to 3d numpy'''
        input_np = ( train_data.values)
        input_np = [np.vstack(item)for item in (input_np)]
        return input_np

    @staticmethod
    def convert_to_one_hot(inputY, C):
        "Convert Y to one hot representation"
        N= inputY.size
        Y=np.zeros((N,C))
        for i in range (0, inputY.size):
            Y[i, int(inputY[i])] = 1
        return Y

    def split_x_y(self,df,class_number=5): 
        '''obtain the entry data, the label data in its different forms'''
        gesture_array=self.gesture_array
        gesture_list =self.class_names
        df = df[df['label'].isin(gesture_array)]
        X,Y = df.loc[:, df.columns != 'label'],df['label']
        Y_class = np.array([gesture_list[x] for x in Y])
        Y_oh    = self.convert_to_one_hot(Y_class,class_number).tolist()
        print(X)
        return X,Y,Y_class,Y_oh
    
    def resample_signal(self,resample_length): 
        '''resampling the data using the sig.resample function'''
        df = self.df
        df =df.drop(columns=self.meta_columns)
        for column_name in df: 
            column = df[column_name]
            column = column.values
            new_column = []
            for  signal in column:
                resampled_signal= sig.resample(signal,resample_length)
                new_column.append(resampled_signal)
            self.df[column_name] = new_column

    @staticmethod
    def standarize_signal(x): 
        return (x-np.mean(x))/np.std(x)
    @staticmethod
    def mask_signal(sig, mask): 
        new_sig = [sig[i] for i in mask]
        new_sig = np.vstack(new_sig)
        return new_sig
    def standarize_df(self):
        df = self.df.copy(deep=True)
        df = df.drop(columns=self.meta_columns)
        for column_name  in df: 
            self.df.loc[:,column_name] = df[column_name].apply(lambda x:DataLoader.standarize_signal(x))

############## Energie & velocity calculations #######################
    @staticmethod
    def energy_vel(sig):
        '''calculat the velocty of a signal as well as the energy of the this velocity, this is a static method'''
        sig = sig[0]
        energy_vel_arr = []
        for feature_signal in sig: 
            length_signal = feature_signal.shape[0]
            line = np.arange(length_signal)
            samples = [Sample(feature_signal[i],line[i]) for i in range (0,feature_signal.shape[0])]  
            tracker = AlphaBetaFilter(samples[0], alpha=0.95, beta=0.1, velocity=0.0)  # initiate a tracker
            for sample in samples[1:]:
                tracker.add_sample(sample)
            vel = tracker.velocity_list
            vel = np.array(vel)
            energy_vel = np.sum(vel*vel)    
            energy_vel_arr.append(energy_vel)
        return energy_vel_arr

    @staticmethod
    def  calculate_signal_velocity(sig,alpha,beta): 
        '''calculate the signal velocity for a signal sig 
        this is also a static method'''
        length_signal = sig.shape[0]
        line = np.arange(length_signal)
        samples = [Sample(sig[i],line[i]) for i in range (0,sig.shape[0])]  
        tracker = AlphaBetaFilter(samples[0], alpha=alpha, beta = beta, velocity=0.0)  # initiate a tracker
        for sample in samples[1:]:
            tracker.add_sample(sample)
        vel = tracker.velocity_list
        vel = np.array(vel)
        return vel 
    
    def calculate_gesture_velocity(self,alpha,beta,keep_signal=True): 
        '''calculate the velocity for all features'''
        df = self.df
        df = df.drop(columns=self.meta_columns)
        for column_name  in df: 
            column = df[column_name]
            column = column.values
            new_column = []
            for  signal in column:
                signal_vel= DataLoader.calculate_signal_velocity(signal,alpha,beta)
                new_column.append(signal_vel)
            if  (keep_signal!= True):
                self.df = self.df.drop(columns=[column_name])
            self.df[column_name+'vel'] = new_column
    
    def calculate_filter_data_velocity_velocity(self,array_to_filter,alpha,beta,keep_signal=True): 
        ''''''
        df = self.df
        df = df[array_to_filter]
        for column_name  in df: 
            column = df[column_name]
            column = column.values
            new_column = []
            for  signal in column:
                signal = np.array(signal)
                signal_vel= self.calculate_signal_velocity(signal,alpha,beta)
                new_column.append(signal_vel)
            if  (keep_signal!= True):
                self.df = self.df.drop(columns=[column_name])
            self.df[column_name+'vel'] = new_column
    
    
    def get_energy_right_left_df(self):
        '''calcucate the energy independently for left and right side''' 
        df = self._get_dataframe()

        energy_ARR,energy_ARR_vel = [] , []
        for i in range (0,len(df)):
            diver = df.iloc[i]
            name= diver['Name']
            hand = diver['hands']
            label = diver['label']
            signal_id= diver['signal_id']
            flipped = diver['original']
            oscillation = diver['oscillations']
            sig,_,_ =  self.extract_row(name,signal_id,with_extra = False,copy=flipped)
            energy, sig_normed = self.feat_energy(sig)
            row = [signal_id] + energy +[label,name,hand,oscillation,flipped]
            row_vel = self.energy_vel(sig)
            energy_ARR.append(row)
            energy_ARR_vel.append(row_vel)
        colums_vel = ['velocity_energy_sig'+str(i) for i in self.all_features] 
        energy_ARR = np.array(energy_ARR,dtype=object)
        energy_ARR_vell = np.array(energy_ARR_vel,dtype=object)
        energy_df = pd.DataFrame(energy_ARR, columns = ([ 'signal_id']+self.all_features+['label', 'Name','hands','oscillations','original'] ) )
        df_copy_energy = pd.DataFrame(energy_ARR_vel,columns=colums_vel)
        energy_df = energy_df.join(df_copy_energy)
        print(energy_df.columns)
        energy_df['energy_right_side']=energy_df[self.right_side_feature].sum(axis=1)
        energy_df['energy_left_side']=energy_df[self.left_side_feature].sum(axis=1)
        energy_df['energy_vel_right']=energy_df[['velocity_energy_sig' +  i for i in self.right_side_feature ]].sum(axis=1)
        energy_df['energy_vel_left']=energy_df[['velocity_energy_sig' +  i for i in self.left_side_feature ]].sum(axis=1)
        energy_df['h']  = np.where(energy_df['energy_right_side'].values > energy_df['energy_left_side'].values,2,1) 
        energy_df['x1'] = np.where(energy_df['energy_right_side'].values > energy_df['energy_left_side'].values,(energy_df['energy_right_side']- energy_df['energy_left_side'])/ energy_df['energy_right_side']  ,(energy_df['energy_left_side']- energy_df['energy_right_side'])/ energy_df['energy_left_side'] ) 
        energy_df['x2'] = np.where(energy_df['energy_vel_right'].values > energy_df['energy_vel_left'].values,(energy_df['energy_vel_right']- energy_df['energy_vel_left'])/ energy_df['energy_vel_right']  ,(energy_df['energy_vel_left']- energy_df['energy_vel_right'])/ energy_df['energy_vel_left'] ) 
        return energy_df
    
        
    def feat_energy(self,sig): 
        '''returns the energy of each feature signal'''
        sig = sig[0]
        sig = [sig[i] - sig[i,0] for i in range(0,len(sig)) ] 
        energy = [np.sum(sig[i]*sig[i]) for i in range(0,len(sig))]
        return energy, sig
    
    def split_right_left_angles(self):
        '''divide the data into left and right body data'''
        df = self.df
        right_signal =df.drop(columns=self.left_side_feature)
        left_signal =df.drop(columns=self.right_side_feature)
        return right_signal,left_signal
    
    def seperate_additional_data(self,df=''): 
        if not isinstance(df, pd.DataFrame): 
            df = self.df 
        '''get df data into the shape we used to have before, signal + label, 
        all the metadata is loaded into another df'''
        meta_data = df[self.meta_columns]
        return meta_data,df
    
 