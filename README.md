# Mo2E
Download hyper-kvasir dataset from https://datasets.simula.no/hyper-kvasir/ , download : hyper-kvasir-labeled-images.zip dataset and use this.

After downloading the data split the training set into 4 random splits in the ratio (80:20) (train:test).

all networks needs to train for different values of alpha given in run_experiments_resnxt_endo.sh

After training the models you can finally combine it by using the python script results_combine.py, by using trained networks.

Eyepac dataset can be downloaded from the kaggle directly: https://www.kaggle.com/datasets/agaldran/eyepacs

