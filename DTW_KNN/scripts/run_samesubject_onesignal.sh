#!/bin/bash
cd ../
yaml_file='./yaml/DTW_samediver.yml'
#directory in which the data is stored

# The gesture data type, can be either no sep for data without svm seperation, duo for only gestures with two arms or mono for only gestures with one arm
data_type='nosep/'
#the corresponding DTW window size
window_size=2
#number of neighbors in KNN 
neighbors=10 
subject='subject3' 
signal_number=5
output='./../output/repeatability/'

if [ ! -d "$dir" ]; then
    if [ ! -e "$dir" ]; then
        # Create the directory
        mkdir "$output"
    fi
fi

python3 dtw_samediver_script.py --window $window_size --neighbors $neighbors --subject $subject --data_directory $data_type --yaml $yaml_file --prefix $output --signal_ith $signal_number
