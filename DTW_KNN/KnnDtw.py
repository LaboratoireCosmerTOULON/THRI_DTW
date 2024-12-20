
from asyncio.subprocess import SubprocessStreamProtocol
from cmath import sin
from curses import meta
from http.client import NO_CONTENT
from inspect import trace
import sys
from tabnanny import verbose
from xml.sax.handler import feature_string_interning
import numpy as np
from scipy.stats import mode
from scipy.spatial.distance import squareform
from sklearn import semi_supervised
from sklearn.metrics import euclidean_distances
from utils.ProgressBar import ProgressBar
import pandas as pd


class KnnDtw(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays

    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN
        
    max_warping_window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function
            
    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    """
    
    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1,distance_type='normal',class_names=[], info_type ='normal',wrapping_calculation = False,prefix = '',rho=1,penality =0):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step
        self.distance_type = distance_type
        self.prefix = prefix
        self.info_type =info_type
        self.class_names = class_names
        self.wrapping_calculation = wrapping_calculation
        self.rho=1
        self.penality = penality
    
    def fit(self, x, l,metadata = 0,metalabel = 0 ):
        """Fit the model using x as training data and l as class labels
        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer 
        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """
        self.x = x
        self.l = l
        self.metadata = metadata
        self.metalabel =metalabel
    
    ############## functions added for better distances definition#####
    def slicer (a): 
        return [a[i*3:i*3+3]for i in range(len(a)//3)]
    def angle_distance_1_angle(a,b): 
        angle = a-b 
        angle = ((angle+ 3*np.pi )%(2*np.pi) )- np.pi
        return angle

    def eucl(a,b): 
        return np.sqrt(np.sum(np.square(a-b)))
    def euclid_distance_sep(a,b): 
        a,b = KnnDtw.slicer(a),KnnDtw.slicer(b)
        return np.sum([KnnDtw.eucl(a[i],b[i])for i in range(len(a))]) 
    def angle_distance(a,b):    
        diff = [ abs(KnnDtw.angle_distance_1_angle(a[i],b[i]) )  for i in range(len(a))]
        return np.sum(diff),diff
    def norm1_distance(a,b): 
        return np.sum(abs(np.array(a) - np.array(b))),abs(np.array(a) - np.array(b))
    def norm2_distance(a,b): 
        return np.sqrt(np.sum([(a[i]-b[i])**2 for i in range(len(a))]) ),[(a[i]-b[i])**2 for i in range(len(a))]
    
    def distance(self,a,b): 
        if self.distance_type=='norm1': 
            return KnnDtw.norm1_distance(a,b)
        if self.distance_type=='norm2': 
            return KnnDtw.norm2_distance(a,b)
        if self.distance_type== 'normal': 
            return abs(a-b)
        if self.distance_type== 'mixed-euclidean':
            return np.sqrt(np.sum(np.square(a-b)))
        if self.distance_type== 'sep-euclidean':
            return KnnDtw.euclid_distance_sep(a,b)
        if self.distance_type =='angle': 
            return KnnDtw.angle_distance(a,b)
        
    def _dtw_distance(self, ts_a, ts_b, d = lambda x,y: abs(x-y)):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared
        
        d : DistanceMetric object (default = abs(x-y))i
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function
        
        Returns
        -------
        DTW distance between A and B
        """

        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a.T), np.array(ts_b.T)
        M, N, F= ts_a.shape[0], ts_b.shape[0], ts_b.shape[1]
        cost = sys.maxsize * np.ones((M+1, N+1))
        cost_meta = sys.maxsize * np.ones((M+1, N+1,F))
        traceback_matrix = 5*np.ones((M,N))

        features_contribution = []
        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])[0]
        cost_meta[0, 0] = d(ts_a[0], ts_b[0])[1]
        for i in range(1, M+1):
            cost[i, 0] = cost[i-1, 0] + d(ts_a[i-1], ts_b[0])[0]
            cost_meta[i,0,:] = d(ts_a[i-1], ts_b[0])[1]

        for j in range(1, N+1):
            cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j-1])[0]
            cost_meta[0,j,:] = d(ts_a[0], ts_b[j-1])[1]
        # Populate rest of cost matrix within window
        for i in range(M):
            for j in range(max(0, i - self.max_warping_window),
                            min(N, i + self.max_warping_window)):
                choices = cost[i, j],  cost[i, j+1]+self.penality,cost[i+1, j]+self.penality
                cost[i+1, j+1] = min(choices) + d(ts_a[i], ts_b[j])[0]
                cost_meta[i+1, j+1,:] =  d(ts_a[i], ts_b[j])[1]
                i_choice = np.argmin(choices)
                traceback_matrix[i,j] = i_choice
        # Return DTW distance given window
        if self.wrapping_calculation== True : 
            i = M-1
            j = N-1
            path = [(i,j)]
            #features_contribution.append(cost_meta[i,j,:])
            while i>0 or j>0: 
                tb_type =traceback_matrix[i,j]
                features_contribution.append(cost_meta[i,j,:])
                
                if i == 0 :
                    j = j-1 
                elif j ==0 :
                    i =i-1
                elif tb_type == 0:
                    # Match
                    i = i - 1
                    j = j - 1
                elif tb_type == 1:
                    # Insertion
                    i = i - 1
                elif tb_type == 2:
                    # Deletion
                    j = j - 1

                elif tb_type == 5: 
                    if i>0 :
                        i = i -1 
                    if j>0 : 
                        j=j-1
                path.append((i, j))
            path.append((0,0))
            features_contribution.append(cost_meta[0,0,:])
            features_contribution = np.array(features_contribution)
            features_contribution = np.sum(features_contribution,axis=0)
            return cost[1:,1:],traceback_matrix, path,features_contribution
        else: 
            return cost[-1, -1]
            
    
    def _dist_matrix(self, x, y,multivar=False):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure
        
        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
        
        y : array of shape [n_samples, n_timepoints]
        
        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """
    
        # Compute the distance matrix        
        dm_count = 0
        
        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if (isinstance(x,list) and isinstance(y,list)): 
            print('LIST IMPLEMENTATION')
            x_s = len(x)
            y_s = len(y)
            dm = np.zeros(((x_s),(y_s)))
            #print('shape used', x_s,y_s)
            dm_size = (x_s*y_s)
            p = ProgressBar(dm_size)
            for i in range(0, x_s,self.subsample_step):
                for j in range(0, y_s,self.subsample_step):
                    if (multivar ==True): 
                        #print(x[i].shape,y[j].shape)
                        dm[i,j] = self._dtw_distance(x[i],y[j], d = lambda a,b :KnnDtw.distance(self,a,b) ) 
                    else :     dm[i,j] = self._dtw_distance(x[i],y[j])
                    dm_count += 1
                    p.animate(dm_count)
            
            return dm  
        elif(np.array_equal(x, y)):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)
            
            p = ProgressBar(np.shape(dm)[0])
            
            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]-1):
                    if (multivar ==True): 
                        dm[dm_count] = self._dtw_distance(x[i, ::self.subsample_step],
                                                        y[j, ::self.subsample_step],d = lambda a,b :KnnDtw.distance(self,a,b))
                    else :     dm[dm_count] = self._dtw_distance(x[i, ::self.subsample_step],y[j, ::self.subsample_step])
                    dm_count += 1
                    p.animate(dm_count)
            
            # Convert to squareform
            dm = squareform(dm)
            return dm
        
        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            #print(x_s,y_s)
            dm = np.zeros(((x_s[0]), (y_s[0]))) 
            dm_size = (x_s[0])*(y_s[0])
            
            p = ProgressBar(dm_size)
            print('shapes used' ,x_s[0],y_s[0])
            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    if (multivar ==True): 
                        dm[i,j] = self._dtw_distance(x[i, ::self.subsample_step],y[j, ::self.subsample_step],d = lambda a,b :KnnDtw.distance(self,a,b) )
                    else :     dm[i,j] = self._dtw_distance(x[i, ::self.subsample_step],y[j, ::self.subsample_step])
                    # Update progress bar
                    dm_count += 1
                    p.animate(dm_count)
        
            return dm
    def  _dist_matrix_verbose(self,x,ith):
            verbose_metadata = self.metadata
            dm_count = 0 
            y = x[ith]
            meta_data_y = self.metadata.iloc[ith]
            x_s = len(x)
            y_s = 1
            dm = np.zeros((x_s))
            feat_num = x[0].shape[0]
            feat_contribs = np.zeros((x_s,feat_num))
            dm_size = (x_s)
            p = ProgressBar(dm_size)
            for i in range(0, x_s):
            #for i in range(0, 1):
                    cost_mat,nana,susu,feat_contrib = self._dtw_distance(x[i],y, d = lambda a,b :KnnDtw.distance(self,a,b)) 
                    dm[i]=cost_mat[-1,-1]
                    feat_contribs[i,:] = feat_contrib
                    dm_count += 1
                    p.animate(dm_count)
            feat_contribs = np.array(feat_contribs)
            print(dm.shape, verbose_metadata.size,feat_contribs.shape)
            for i in range (0,feat_num): 
                col_name = 'contrib_angle'+str(i)
                verbose_metadata[col_name] = feat_contribs[:,i]  

            verbose_metadata = verbose_metadata.assign(weight=dm)
            print(self.l)
            verbose_metadata = verbose_metadata.assign(label=self.metalabel)
            return dm,verbose_metadata
    def  _dist_matrix_verbose_sep(self,x,test,test_meta):
            verbose_metadata = self.metadata
            dm_count = 0 
            y = test
            meta_data_y = test_meta
            x_s = len(x)
            y_s = 1
            dm = np.zeros((x_s))
            feat_num = x[0].shape[0]
            feat_contribs = np.zeros((x_s,feat_num))
            print('shape used', x_s,y_s)
            dm_size = (x_s)
            p = ProgressBar(dm_size)
            for i in range(0, x_s):
            #for i in range(0, 1):
                   # print(x[i].shape,y.shape)
                    cost_mat,nana,susu,feat_contrib = self._dtw_distance(x[i],y, d = lambda a,b :KnnDtw.distance(self,a,b)) 
                    dm[i]=cost_mat[-1,-1]
                    feat_contribs[i,:] = feat_contrib
                    dm_count += 1
                    p.animate(dm_count)
            feat_contribs = np.array(feat_contribs)
            print(dm.shape, verbose_metadata.shape,feat_contribs.shape)
            for i in range (0,feat_num): 
                col_name = 'contrib_angle'+str(i)
                print(i)
                verbose_metadata[col_name] = feat_contribs[:,i]  
            verbose_metadata = verbose_metadata.assign(weight=dm)
            print(self.l)
            verbose_metadata = verbose_metadata.assign(label=self.metalabel)
            return dm,verbose_metadata
    
    def predict(self, x,samesubject=True,multivar=True):
        """Predict the class labels or probability estimates for 
        the provided data

        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Array containing the testing data set to be classified
            print(i,j)
        Returns
        -------
        2 arrays representing:
            (1) the predicted class labels 
            (2) the knn label count probability
        """
        dm = self._dist_matrix(x, self.x,multivar)
        # Identify the k nearest neighbors argsort return the indices not the sorted array 
        knn_idx = dm.argsort()[:, :self.n_neighbors]
        if samesubject: 
            knn_idx = dm.argsort()[1:self.n_neighbors+1]
        # Identify k nearest labels
        knn_labels = self.l[knn_idx]
        # Model Label
        mode_data = mode(knn_labels, axis=1)
        #gives the mode 
        mode_label = mode_data[0]
        #mode_data[1] is the mode count 
        mode_proba = mode_data[1]/self.n_neighbors
        print('Prediction with ', self.n_neighbors, 'neighbors' ,mode_label[0],\
               self.class_names[mode_label[0]], 'Confidence: ',mode_proba)
        return mode_label.ravel(), mode_proba.ravel()
    

    def predict_verbose(self,ith=5,multivar = True,suffix='',samesubject=False): 
        prefix = self.prefix 
        meta_data_y = self.metadata.iloc[ith]
        print (meta_data_y)
        signal_id = meta_data_y['signal_id']
        diver_name= meta_data_y['Name']
        unique_id=diver_name+str(signal_id)
        dm,verbose_meta = self._dist_matrix_verbose(self.x,ith=ith)
        knn_idx = dm.argsort()[:self.n_neighbors]
        if samesubject: 
            knn_idx = dm.argsort()[1:self.n_neighbors+1]
        knn_labels = self.l[knn_idx]
        print('knn_idx', knn_idx)
        print('knn_lables', knn_labels)
        mode_data = mode(knn_labels, axis=0)
        mode_label = mode_data[0]
        mode_proba = mode_data[1]/self.n_neighbors
        print('Prediction with ', self.n_neighbors, 'neighbors' ,mode_label[0],\
               self.class_names[mode_label[0]], 'Confidence: ',mode_proba) 
        sorted_meta = verbose_meta.sort_values(by='weight')
        print('saved to : ', prefix+unique_id+suffix)
        sorted_meta.to_csv(prefix+unique_id+suffix+'.csv')
        return mode_label.ravel(), mode_proba.ravel(),knn_idx,dm ,sorted_meta
    
    def predict_verbose_sep(self,test, test_meta, multivar = True,suffix='d'): 
        prefix = self.prefix 
        meta_data_y = test_meta
        print (meta_data_y)
        signal_id = meta_data_y['signal_id']
        dm,verbose_meta = self._dist_matrix_verbose_sep(self.x,test,test_meta)
        knn_idx = dm.argsort()[:self.n_neighbors]
        knn_labels = self.l[knn_idx]
        print('knn_idx', knn_idx)
        print('knn_lables', knn_labels)
        mode_data = mode(knn_labels, axis=0)
        mode_label = mode_data[0]
        mode_proba = mode_data[1]/self.n_neighbors
        print('Prediction with ', self.n_neighbors, 'neighbors' ,mode_label[0],\
               self.class_names[mode_label[0]], 'Confidence: ',mode_proba) 
        sorted_meta = verbose_meta.sort_values(by='weight')
        print(sorted_meta)
        diver_name= meta_data_y['Name']
        unique_id=str(diver_name)+str(signal_id)
        print(suffix)
        print(prefix+unique_id+suffix+'.csv')
        sorted_meta.to_csv(prefix+unique_id+suffix+'.csv')
        return mode_label.ravel(), mode_proba.ravel(),knn_idx,dm ,sorted_meta
    def wrap_signal(self,signal_with_right_size,signal_to_warp,path):
        warpped_signal = np.zeros((signal_with_right_size.shape))
        for couple in path : 
            index_i = couple[0]
            index_j = couple[1]
            warpped_signal[:,index_i] = signal_to_warp[:,index_j]
        return warpped_signal
