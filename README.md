# LPI-CSFFR
LPI-MFC
Predicting lncRNA-protein interactions by combining serial fusion with feature reuse 

we proposed a computational method LPI-MFC to predict lncRNA-protein interactions from sequences, secondary structure, and physicochemical property, which made use of deep learning CNN and further improve its performance using feature reuse strategy. 

Dependency:

python 3.6.5

Requirment:

h5py==2.10.0

Keras==2.1.0

matplotlib== 3.2.2

numpy ==1.14.5

tensorflow==1.10.0

scikit-learn== 0.23.2



Before using main.py, you need to adopt Pse-in-One-2.0 tool to obtain the vectors of physicochemical properties by python 2.7.

Take dataset RPI1460 as an example: 

python pse.py  ./data/sequence/RPI1460_rdrna_seq.txt  RNA PC-PseDNC-General -all_index ./data/pse/RPI1460_rdrna_pse.txt

python pse.py  ./data/sequence/RPI1460_rdprotein_seq.txt  Protein PC-PseAAC-General -all_index ./data/pse/RPI1460_rdprotein_pse.txt

Usage: 

python main.py -d 'RPI1460' -sample 'random' -mode 'cnnsep_denseblock'

where 'RPI1460' is lncRNA-protein interaction dataset, 'random' means the negative samples being randomly selected, 'cnnsep_denseblock' is a serial fusion framework MFC_SER, and LPI-MFC will do 5-fold cross-validation for it. you can also choose the other dataset 'RPI1807', the other sample 'swrandom', and mode 'cnn_denseblock'.



Reference:

LPI-CSFFR: Combining Serial Fusion with Feature Reuse for Predicting LncRNA-Protein Interactions 

Contact: tanjianjun@bjut.edu.cn
