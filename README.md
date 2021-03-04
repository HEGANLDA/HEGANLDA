# HEGANLDA

#### Source code for paper " HEGANLDA：a computational model for potential lncRNA-disease association prediction based on multiple heterogeneous networks”.

## Get started

#### Evironment Setting
##### &emsp; * Python version >= 3.6
##### &emsp; * PyTorch version >= 1.4.0
##### &emsp; * TensorFlow version >= 1.14

* ### 1.Datasets
##### &emsp;&emsp; HEGANLDA collected six networks data (lncRNA-disease, lncRNA-miRNA, disease-miRNA, lncRNA-lncRNA, disease-disease, and miRNA-miRNA) to construct a lncRNA-miRNA-disease heterogeneous network. 

* ### 2.Nodevector
#####  &emsp; &emsp; HeGAN (B. Hu et al., 2019) algorithm is employed to map all the nodes in the lncRNA-miRNA-disease heterogeneous network into low-dimensional vectors.
#### &emsp;&emsp; * Config/Usage
##### &emsp;&emsp;&emsp;&emsp; Input parameter : 
##### &emsp;&emsp;&emsp;&emsp;&emsp; **python train.py -m HeGAN -d LD**
##### &emsp;&emsp;&emsp;&emsp; If you want to train your own dataset, create the file (./dataset/LD/edge.txt) and the format is as follows:

![edge.png](https://github.com/HEGANLDA/HEGANLDA/blob/main/images/edge.png)


##### &emsp;&emsp; &emsp;&emsp;And the input graph is directed and the undirected needs to be transformed into directed graph.
#### &emsp;&emsp; * Modle Setup
##### &emsp;&emsp;&emsp;&emsp; The model parameter could be modified in the file ( ./src/config.ini ). 
##### &emsp;&emsp;&emsp;&emsp; Common parameter :
##### &emsp;&emsp;&emsp;&emsp;&emsp; * dim : dimension of output
##### &emsp;&emsp;&emsp;&emsp;&emsp; * epoch : the number of iterations
##### &emsp;&emsp;&emsp;&emsp; Output :
##### &emsp;&emsp;&emsp;&emsp;&emsp; The results are stored in the file (./output/embedding/HeGAN).  

* ### 3. lncRNA-disease_association_vector
#####  &emsp; &emsp; Five methods are implemented to gain the lncRNA-disease vectors.<br>&emsp; &emsp; For the LDVCHN method, each node was assigned to a corresponding weight according to the structural information of the heterogeneous network. 

* ### 4. Classifier
#####  &emsp; &emsp; The ROC curve and AUC value of the model with five different classifiers are compared based on vector. And the Xgboost classifier is adopted to predict potential lncRNA-disease associations.<br>&emsp; &emsp; The LDVCHN method is chosen to calculate the vector based on the AUC value.<br>&emsp; &emsp; The AUC values of the HEGANLDA model were compared when λ is set to 32, 64, and 128 through 10-fold cross validation. And the value of λ is set to 128.

* ### 5.Predict
#####  &emsp; &emsp; Firstly, the vectors of all unknown lncRNA-disease associations are calculated. Then, XGBoost classifier is used to predict the possibility of association between node pairs





