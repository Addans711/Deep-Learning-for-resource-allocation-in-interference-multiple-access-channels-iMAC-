Please first install the requirement in the python environment

First of all, you should generated the data in the MATLAB program according to the paper, 
for instance, you should run the Generate_IMAC_function.m in the run_IMAC_DATA.mlx file 
and change the parameter setting in the mlx file to generated the iMAC data for training
and testing. By the way, you should modify the save(fullfile('C:\Users\11345\Documents\project\TSP-DNN-master\TSP-DNN-master', ...
    sprintf('IMAC_%d_%d_%d_%d_%d.mat', num_BS, num_User, num_H, R, minR_ratio)), 'H', 'X', 'Y', 'WmmseTime'); 
to whatever you want to save your file, otherwise it will be fail.

Then you can get the result in the imac.ipynb program by using that data you have already 
generated.

To increase the number of layer, I have already put the py file in the Github, as you can
see the function_dnn_powercontrol4 is for 4 layers and function_dnn_powercontrol5 is for 
the 5 layers, you can easily change the import patameter if you want.

To change the number of neurons, you can change the parameter in the function_dnn_powercontrol.py
file, you can find the ' n_hidden_- ' which means the neurons in each layer.
