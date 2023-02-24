
# IMI-Driver
##  Overview
The project consists of four main parts

1、Data preprocessing:/Preprocessing

2、Constructing network:/Network

3、Network embedding:/Cancer_MANE

4、Visualization:/Plot

## Prerequisites
Scikit-learn 0.19.1 

Numpy 1.15.4 

Scipy 1.2.0 

Torch 0.4.1 

Python 3.5

Matlab

Compatible with both cuda and cpu devices, depending on the user choice through arg_parser file. Also compatible with python2 and python3.
##  Implementation
step1:Download data. Here we will use the breast cancer (BRCA) data as an example, the data are available for download at XXX.Then put the `data` in `. /Network` and `ceRNA` in `. /Network/ceRNA`

step2:Network construction.

`cd ./Network`

`matlab all_net_demo.m`

step3:Network embedding.

`cd ./Cancer_MANE/attention/Node_Classification`

`python main_Node_Classification_MANE_Attention.py  --dimensions 64  --epochs 50 --nview 5 --cancer BRCA`

