import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

import peptidy as pep

import sklearn
from sklearn.ensemble import RandomForestRegressor


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


def train_model(X,y,n_estimators=100,  criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
                oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None):
    #ja er komt een uitleg wat alles is en ik ga nog selecteren wat relevant is, dit zijn de standaard waarde van de makers van het model
    #Ik kan doordat ik dit tijdens de datacombinatie gaan doen omdat ik daar errors had nu niet de X,y die daaruit komt gebruiken dus die eventuele errors zal ik nog op moeten lossen
    #Welke input heeft de predict functie nodig @Iris

    #n_estimators is het aantal bomen dat je wilt gaan gebruiken, dit lijkt mij relevant
    #criterion, dit is de Loss functie die je gebruikt om op zoek te gaan naar de beste boom, @Iris welke gebruik jij?
    #Max_depth De maximum depth van de boom, je kan dus ook het maximum qua aantal takken voorstellen. Dit is iets anders dan de minimum aantal samples per afsplitsing. Dit is een hyperparameter die we uit zullen moeten gaan testen
    #min_samples_split minimaal aantal samples per split, dit is een hyperparameter die we sws moeten gaan testen
    #min_samples_leaf, minimaal aantal samples die nodig zijn bij een leaf node, dus met hoeveel je uiteindelijk een keus maakt --> ook testen
    #max_features, waordt er gekenen naar het maximaal aantal features die een boom gebruikt om een boom te maken --> redelijk relevant, miss ook voor featureselection
    #bootstrap --> is relevant, moet aanstaan
    #random_state --> kan denk ik wel nuttig zijn tijdens het testen, maar ook half


    #oob_scire --> out of bag, kan relevant miss ook in plaats van crossvalidation
    #max_samples --> kan relevant zijn, maar ik zou dit niet als eerste testen, als je het niet test is dat denk ik ook prima

    #min_weight_fraction_leaf --> is een andere methode van het aantal uiteindelijk samples bepalen, hoeft niet uitgetest te worden
    #max_leaf_nodes wordt gekeken naar hoeveel leafs er maximaal zijn, kan je denk ik beter met adere features doen
    #min_impurity_decrease, de minimale verbetering --> ik denk dat dit heel lastig is, voorkomt miss overfitting, maar ik denk dat dit te veel extra is
    #n_jobs, hij gaat dan meerdere dingen tegelijk doen, is denk ik niet heel relevant
    #verbose, niet heel nuttig geeft je eventueel meer inzicht over hoever die is
    #warm_start --> herbruikt dan de vorige call om beter te fitten, maar ik denk dat wij dit juist niet willen
    #ccp_alpha --> kan gebruikt worden voor overfitten maar is denk ik nu onnodig complex
    #monotonic_cst, je kan beperkingen aan de richting van invloed van features, is denk ik onnodig ingewikkeld
    

    random_forest=sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators,  criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, 
                                                         min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, 
                                                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, 
                                                        oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, 
                                                        ccp_alpha=ccp_alpha, max_samples=max_samples, monotonic_cst=monotonic_cst)
    random_forest.fit(X,y)
    return random_forest


