# DeepUnfolding_WMMSE
This repository contains the entire code for our TWC work "Iterative Algorithm Induced Deep-Unfolding Neural Networks: Precoding Design for Multiuser MIMO Systems", available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&amp;arnumber=9246287 and has been accepted for publication in TWC.

For any reproduce, further research or development, please kindly cite our TWC Journal paper:

`Q. Hu, Y. Cai, Q. Shi, K. Xu, G. Yu, and Z. Ding, “Iterative algorithm induced deep-unfolding neural networks: Precoding design for multiuser MIMO systems,” IEEE Trans. Wireless Commun., to be published.`

# Requirements
The following versions have been tested: Python 3.6 + Tensorflow 1.12. But newer versions should also be fine.

# Introductions
There are three folders: "`DeepUnfolding`", "`Blackbox CNN`", and "`WMMSE`", where "`DeepUnfolding`" corresponds to the proposed deep-unfolding network in our paper, "`Blackbox CNN`" and "`WMMSE`" are benchmarks.

# DeepUnfolding
## Training and Testing
Run the main program "`train_main.py`".

## The introduction of each file
`train_main.py`: Main program that implements the training and testing stages; 

`objective_func.py`: The sum-rate (loss) function; 

`UW_gradient.py`: The gradient of the loss function with respect to U and W in the last layer; 

`UW_conj_gradient.py`: The conjugate gradient of the loss function with respect to U and W in the last layer; 

`generate_channel.py`: Generate the channels; 

`test_model.py`: To test the model. 

# Blackbox CNN
## Data Prepareation
Firstly, we run the file "`generate_data.m`" in the folder "`GenerateData`" to generate the training dataset, which consists of the inputs in the file "`Input_H.csv`" and the labels in the file "`Output_UW.csv`". Then, the two files should be copied into the folder "`DataSet`".
 
## Training and Testing
Run the main program "`Blackbox CNN.py`", which generates four files in the folder "`DataSet`". Finally, we run the file "`test_predict.m`" in the folder "`Test`" to see the sum-rate performance. Note that the file path in "`test_predict.m`" should be modified correspondingly.

# WMMSE 
Run the main program "`WMMSE.py`". 
