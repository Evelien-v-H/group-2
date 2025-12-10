import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

import peptidy as pep
import sklearn

#Openen data:
def open_data(datafile):
    """The files with data need to be imported.
    
    Input: a csv datafile
    
    Output: a dataFrame"""
    df=pd.read_csv(datafile)
    return df


def data_training(datafile):
    """This function splits the dataset in a SMILES array, UNIProt_ID and a affinity score. 
        
        input: a csv file like the given trainingset. The first colom is the SMILES-string, the second colom is the UNIProt_ID and the third 
        Colom is the affinity score
        
        Output: one array with the SMILES, an array with the UNIProt_IDs and an array with the affinityscore"""
    df=open_data(datafile)
    SMILES=df.iloc[:,0].to_numpy()
    UNIProt_ID=df.iloc[:,1].to_numpy()
    affinity=df.iloc[:,2].to_numpy()
    return SMILES,UNIProt_ID,affinity



#Opmerkingen voor het inleveren van de code:
    #RDkit moet nog het stukje aanvullen 2X worden weggehaald
class small_molecule:
    def __init__(self,SMILES):
        self.SMILES=str(SMILES)
        self.molecule=Chem.MolFromSmiles(str(SMILES))


    def rdkit_descriptor(self):
        """This function returns an array with all sorts of descriptors gotten from rdkit
        
        input: self
        
        output an array"""
        dictionary=Descriptors.CalcMolDescriptors(self.molecule, missingVal=None, silent=True)
        array = np.array(list(dictionary.values()), dtype=float)
        return array


    
class protein:
    def __init__(self, uniprot_id, document):
        self.uniprot_id = uniprot_id
        self.document=document

    def extract_sequence(self):
        """no input but the document earlier made shoudl be in the format of:
        first the uniprot-id, second the proteinacronym and last the one
        letter code sequence of the protein, split by comma's. the first
        line is to tell, which column is which. the output is a dictionary,
        with as keys the uniprotid and as value the one letter code 
        sequence of the protein."""
        file=open(self.document) #document needs to be in the right format
        lines=file.readlines()
        lines.pop(0)
        file.close()
        uniprot_dict={}

        for line in lines: 
            uniprot_id,protein_acronym,protein_sequence=line.split(",")
            uniprot_id=uniprot_id.replace('"','')
            protein_sequence=protein_sequence.strip().replace('"','')
            uniprot_dict[uniprot_id]=protein_sequence

        return uniprot_dict

    def uniprot2sequence(self):
        """no input, returns a string with the protein one letter code sequence"""
        uniprot_dict=self.extract_sequence()
        sequence=uniprot_dict[self.uniprot_id]
        return sequence #returns one letter code sequence of the protein
    
    def sequence2onehot(self,sequence):
        """input is the sequence of a protein in one letter code. Returns a 
        list with the encoded peptide sequence represented as one-hot encoded
        vector."""
        onehot=pep.one_hot_encoding(sequence,822)
        return onehot
    
    def extract_features(self, sequence):
        """extracts the protein features from the amino acid sequence using peptidy. 
        Returns a list of all numerical features from peptidy of this protein"""
        peptidy_features_dict = pep.compute_descriptors(sequence, descriptor_names=None, pH=7)
        peptidy_features_dict.pop('molecular_formula')          #This is the only non-numerical feature and is not useful
        peptidy_features_list = peptidy_features_dict.values()
        return peptidy_features_list


def splittingdata(X_train, y_train, percentage):
    """This function splits the data randomly into training data set and a validation
    data set. These training and validation set are returned as a tuple of 2 tuples
    as (X_training,y_training),(X_validation, y_validation). It splits the the data in
    two with the percentage to determine how big training data set is. Percentage is
    a float between 1 and 0."""
    import numpy as np
    import random

    #calculates trainingsize with percentage
    samples, features= X_train.shape
    training_size=int(percentage*samples)

    #permutation makes a random order so data
    #is split randomly.
    permutation=np.random.permutation(samples)

    #shuffles data with permutation
    X_shuffled=X_train[permutation]
    y_shuffled=y_train[permutation]

    #makes the training and validation sets
    X_training=X_shuffled[:training_size]
    X_validation=X_shuffled[training_size:]
    y_training=y_shuffled[:training_size]
    y_validation=y_shuffled[training_size:]

    training=(X_training,y_training)
    validation=(X_validation,y_validation)

    return training,validation

def set_scaling(X):
    """makes the scaler, from given data set X. the scaler used
    is a minmax scaler. it returns a object with a fixed scaler"""
    scaler=sklearn.preprocessing.MinMaxScaler()
    return scaler.fix(X)

def data_scaling(scaler, X):
    """transforms data from fixed scalar. input is the fixed scaler
    and the data that need to be scaled. the output is th scaled data"""
    return scaler.transform(X)

def RF_fitting(X_train, y_train):
    """fits a random forest to a X_train and a y_train. input is a
    dataset with X_train and y_train in an array. output is the fitted
    model of the randomforest."""
    model= sklearn.ensemble.RandomForestRegressor()
    return model.fit(X_train,y_train)

def RF_predict(model, X_test):
    """uses a defined model to predict the y values of X_test. input
    is an array X_test and the defined model. output is an array of 
    the predicted y values"""
    return model.predict(X_test)

def RF_error(model, X_test, y_test):
    """uses R2 to calculate the error of the model. input is a defined
    model, an array X_test and an array y_test. the output is a error
    as a float."""
    return model.score(X_test,y_test)