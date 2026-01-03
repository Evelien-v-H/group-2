#Dit is het definitieve document, hier mag alleen code in die met pep 8 gestructureerd is en die in het definitieve document moet komen

import pandas as pd

def open_data(datafile):
    """The files with data need to be imported.
    
    Parameters: a csv datafile
    
    Returns: a dataFrame"""
    dataframe=pd.read_csv(datafile)
    return dataframe

def data_to_SMILES_UNIProt_ID(datafile):
    """This function splits the dataset in a SMILES array and UNIProt_ID. In case the data file is an training set, it returns also the affinityscore 
        
        Parameters: a csv file like the given training or testset with 3 coloms. (trainset: first colom:SMILES, second colom:UNIProt_ID, third colom: affinity)
        (testset: first colom:numbers, second colom:SMILES, third colom:UNIProt_ID)
    
        Returns: one array with the SMILES, an array with the UNIProt_IDs and an array with the affinityscore. If it is an testset the affinityscore is returned 
        with 'unknown affinity'"""
    df=open_data(datafile)
    colom_1=df.iloc[:,0].to_numpy()
    colom_2=df.iloc[:,1].to_numpy()
    colom_3=df.iloc[:,2].to_numpy()

    if colom_1[0]==0:
        #This if statements is only correct for the format of the given testdocument. This means if the testdocument is build different this function can not be used.
        SMILES=colom_2
        UNIProt_ID=colom_3
        affinity='unknown affinity'
    
    else:
        #This if statements is only correct for the format of the given trainingdocument. This means if the trainingdocument is build different this function can not be used.
        SMILES=colom_1
        UNIProt_ID=colom_2
        affinity=colom_3

    return SMILES,UNIProt_ID,affinity
