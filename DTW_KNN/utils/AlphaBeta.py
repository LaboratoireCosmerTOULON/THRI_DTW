from scipy import signal
import numpy as np
class Sample:
    def __init__(self, x, t):
        self.location = x
        self.time = t

    def __repr__(self):
        return f"Sample({self.location}, {self.time})"


class AlphaBetaFilter:
    def __init__(self, init_sample, alpha=1, beta=0.1, velocity=1):
        self.alpha = alpha
        self.beta = beta
        self.velocity_list = [velocity]
        self.sample_list = [init_sample]
        self.locations = [init_sample.location]
        self.errors = []
        self.predictions = []

    @property
    def last_sample(self):
        return self.sample_list[-1]

    @property
    def last_velocity(self):
        return self.velocity_list[-1]

    def add_sample(self, s: Sample):
        delta_t = s.time - self.last_sample.time
        expected_location = self.predict(delta_t)
        error = s.location - expected_location
        location = expected_location + self.alpha * error
        v = self.last_velocity + (self.beta / delta_t) * error

        # for debugging and results
        self.velocity_list.append(v)
        self.locations.append(location)
        self.sample_list.append(s)
        self.errors.append(error)

    def predict(self, t):
        prediction = self.last_sample.location + (t * self.last_velocity)

        # for debugging and results
        self.predictions.append(prediction)
        return prediction

class FrequencyFilter:
    def __init__(self,w1, w2,filter_type):
        self.fs = 100
        self.nyq= 0.5*self.fs
        self.w1 = w1 
        self.w2 = w2
        self.type  = filter_type
        self.pad_length = 100
        if self.type == 'band': 
            print(self.w1/self.nyq, self.w2/self.nyq)
            a,b = signal.butter(N = 3, Wn=[self.w1/self.nyq,self.w2/self.nyq], btype = 'band', analog=False)
        elif self.type == 'low': 
            a,b = signal.butter(N = 3, Wn=self.w1/self.nyq, btype = 'low', analog=False)
        else: print('the name of the filter is incorrect')
        self.a = a
        self.b = b
    def filter(self,sig):
        if (len (sig) >50):
            pad_length = self.pad_length
            padded_signal = np.pad(sig, pad_length, mode='reflect')

            # Apply the filter using filtfilt
            filtered_signal = signal.filtfilt(self.a,self.b,padded_signal)
            # Remove the padded regions
            sig = filtered_signal[pad_length:-pad_length]
            
        return sig


