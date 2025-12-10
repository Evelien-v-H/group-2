import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

import peptidy as pep
from peptidy import descriptors
import sklearn

#Openen data:
def open_data(datafile):
    """The files with data need to be imported.
    
    Input: a csv datafile
    
    Output: a dataFrame"""
    df=pd.read_csv(datafile)
    return df


def data_training_splitting(datafile):
    """This function splits the dataset in a SMILES array, UNIProt_ID and a affinity score. 
        
        input: a csv file like the given trainingset. The first colom is the SMILES-string, the second colom is the UNIProt_ID and the third 
        Colom is the affinity score
        
        Output: one array with the SMILES, an array with the UNIProt_IDs and an array with the affinityscore"""
    df=open_data(datafile)
    SMILES=df.iloc[:,0].to_numpy()
    UNIProt_ID=df.iloc[:,1].to_numpy()
    affinity=df.iloc[:,2].to_numpy()
    return SMILES,UNIProt_ID,affinity

def data_test_splitting(datafile):
    """This function splits the dataset in a SMILES array, UNIProt_ID and a affinity score. 
        
        input: a csv file like the given testset. The first colom is the SMILES-string and the second colom is the UNIProt_ID
        
        Output: one array with the SMILES, an array with the UNIProt_IDs and an array with the affinityscore"""
    df=open_data(datafile)
    SMILES=df.iloc[:,0].to_numpy()
    UNIProt_ID=df.iloc[:,1].to_numpy()
    return SMILES,UNIProt_ID

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
        sequence of the protein. (For this project it is the protein_info.csv)"""
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
        peptidy_features_dict = descriptors.compute_descriptors(sequence, descriptor_names=None, pH=7)
        peptidy_features_list = list(peptidy_features_dict.values())
        return peptidy_features_list
    
def combining_all_features_training(datafile):
    """This functions makes an matrix with the descriptors from the ligands and proteins in the file
    
    Input: csv-file with a format of the trainingsset (colom 1:SMILES, colom 2:UNIProt_ID, colom 3:affinity)
    
    Output: matrix (samples*features)
    """
    SMILES,UNIProt_ID,affinity=data_training_splitting(datafile)

    for i in range (len(SMILES)):
        ligand=small_molecule(SMILES[i])
        ligand_features=ligand.rdkit_descriptor()

        peptide=protein(UNIProt_ID[i],'data/protein_info.csv' )
        peptide_features_list=peptide.extract_features(peptide.uniprot2sequence())
        peptide_features=np.array(peptide_features_list)
        all_features=np.concatenate((ligand_features, peptide_features))

        if i==0:
            matrix=all_features
        
        else:
            matrix=np.vstack((matrix,all_features))

    
    return matrix

def combining_all_features_test(datafile):
    """This functions makes an matrix with the descriptors from the ligands and proteins in the file
    
    Input: csv-file with a format of the testset (colom 1:SMILES, colom 2:UNIProt_ID)
    
    Output: matrix (samples*features)
    """
    SMILES,UNIProt_ID=data_test_splitting(datafile)

    for i in range (len(SMILES)):
        ligand=small_molecule(SMILES[i])
        ligand_features=ligand.rdkit_descriptor()

        peptide=protein(UNIProt_ID[i],'data/protein_info.csv' )
        peptide_features_list=peptide.extract_features(peptide.uniprot2sequence())
        peptide_features=np.array(peptide_features_list)
        
        all_features=np.concatenate((ligand_features, peptide_features))


        if i==0:
            matrix=all_features
        
        else:
            matrix=np.vstack(matrix,all_features)
    
    return matrix

print(combining_all_features_training('data/train.csv'))


