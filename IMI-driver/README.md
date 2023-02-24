
# IMI-Driver
##  Overview
The project consists of four main parts

1、Data preprocessing:/Preprocessing

2、Constructing network:/Network

3、Network embedding:/Cancer_MANE

4、Visualization:/Plot

## Prerequisites
scikit-learn 0.19.1 

numpy 1.15.4 

scipy 1.2.0 

torch 0.4.1 

Python 3.5

Compatible with both cuda and cpu devices, depending on the user choice through arg_parser file. Also compatible with python2 and python3.
##  Implementation
step1:Download data. Here we will use the breast cancer (BRCA) data as an example, the data are available for download at XXX.

step2:Network construction.

`cd ./Network`

`matlab all_net_demo.m`

step3:Network embedding.

`cd ./Cancer_MANE/attention/Node_Classification`

`python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer BRCA`

