import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

import peptidy as pep

from peptidy import descriptors
import sklearn
from sklearn.ensemble import RandomForestRegressor


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
    SMILES=df.iloc[:,1].to_numpy()
    UNIProt_ID=df.iloc[:,2].to_numpy()
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
        """no input but the document earlier made should be in the format of:
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


def train_model(X,y,n_estimators=100,  criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
                oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None):
    #ja er komt een uitleg wat alles is en ik ga nog selecteren wat relevant is, dit zijn de standaard waarde van de makers van het model
    #Ik kan doordat ik dit tijdens de datacombinatie gaan doen omdat ik daar errors had nu niet de X,y die daaruit komt gebruiken dus die eventuele errors zal ik nog op moeten lossen

    #n_estimators is het aantal bomen dat je wilt gaan gebruiken, dit lijkt mij relevant
    #criterion, dit is de Loss functie die je gebruikt om op zoek te gaan naar de beste boom,
    #Max_depth De maximum depth van de boom, je kan dus ook het maximum qua aantal takken voorstellen. Dit is iets anders dan de minimum aantal samples per afsplitsing. Dit is een hyperparameter die we uit zullen moeten gaan testen
    #min_samples_split minimaal aantal samples per split, dit is een hyperparameter die we sws moeten gaan testen
    #min_samples_leaf, minimaal aantal samples die nodig zijn bij een leaf node, dus met hoeveel je uiteindelijk een keus maakt --> ook testen
    #max_features, waordt er gekenen naar het maximaal aantal features die een boom gebruikt om een boom te maken --> redelijk relevant, miss ook voor featureselection
    #bootstrap --> is relevant, moet aanstaan
    #random_state --> kan denk ik wel nuttig zijn tijdens het testen, maar ook half


    #oob_scire --> out of bag, kan relevant miss ook in plaats van crossvalidation
    #max_samples --> kan relevant zijn, maar ik zou dit niet als eerste testen, als je het niet test is dat denk ik ook prima

    #min_weight_fraction_leaf --> is een andere methode van het aantal uiteindelijk samples bepalen, hoeft niet uitgetest te worden
    #max_leaf_nodes wordt gekeken naar hoeveel leafs er maximaal zijn, kan je denk ik beter met andere features doen
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

def extract_features(self, sequence):
    """extracts the protein features from the amino acid sequence using peptidy. 
    Returns a list of all numerical features from peptidy of this protein"""    
    peptidy_features_dict = descriptors.compute_descriptors(sequence, descriptor_names=None, pH=7)
    peptidy_features_list = list(peptidy_features_dict.values())
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
    n_samples, n_features= X_train.shape
    training_size=int(percentage*n_samples)

    #permutation makes a random order so data
    #is split randomly.
    permutation=np.random.permutation(n_samples)

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
    
def combining_all_features_training(datafile):
    """This functions makes an matrix with the descriptors from the ligands and proteins in the file
    
    Input: csv-file with a format of the training set (colom 1:SMILES, colom 2:UNIProt_ID, colom 3:affinity)
    
    Output: matrix (n_samples*n_features) and affinity (np.array of length n_samples)
    """
    SMILES,UNIProt_ID,affinity=data_training_splitting(datafile)
    for i in range(len(SMILES)):
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
    print(SMILES,UNIProt_ID)
    for i in range(len(SMILES)):
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

def fit_PCA(X, n_components=None):
    """performs a PCA on the data in X (np.array) and 
    returns the features transformed onto the principal component feature space as X_reduced (np.array)"""
    X_reduced = sklearn.decomposition.PCA(n_components=n_components).fit_transform(X)
    return X_reduced

def plot_PCA(X_reduced, y, component1, component2):
    """makes a PCA scatterplot with on the horizontal axis component1 and component2. 
    Parameters: 
        X_reduced (np.array): all data transformed onto the feature space of the principle components.
        component1 (int): index of principal component to be plotted on horizontal axis
        component2 (int): index of principal component to be plotted on vertical axis
     """
    fig = plt.figure()
    ax = fig.add_subplot()
    scatter = ax.scatter(X_reduced[:,component1], X_reduced[:,component2], c=y)
    return scatter

#Onderstaande functie is voor ons eigen gebruik, voor als we het complete model willen gaan testen
def run_model():
    """runs the complete model, including as many functions we have right now as possible"""
    X,y = combining_all_features_training('data/train.csv')
    train_set, validation_set = splittingdata(X, y, 0.3)      #splits 30% of the data to the validation set, which is reserved for evaluation of the final model
    X_train_raw, y_train = train_set
    scaler = set_scaling(X_train_raw)
    X_train_scaled = data_scaling(scaler, X_train_raw)
    #Hier moet de data cleaning functie komen (Evelien)
    X_reduced_pca = fit_PCA(X_train_scaled)
    scatterplot = plot_PCA(X_reduced_pca, y_train, 0, 1)
    for data_source in [X_train_raw, X_train_scaled, X_reduced_pca]:    #Hier moet nog X_train_cleaned bij (Evelien)
        highest_cv_score = 0
        clf = sklearn.ensemble.RandomForestRegressor()      #hier moeten nog hyperparameters in
        mean_cv_score = sklearn.model_selection.cross_val_score(clf, data_source, y_train, cv=5).mean()       #vinden we cv=5 goed?
        if mean_cv_score > highest_cv_score:
            highest_cv_score = mean_cv_score
            best_data_source = data_source
    return best_data_source



def data_cleaning(data):
    """Input data matrix"""
    
    for i in range(data.shape[1]):
        print('b')
        for j in range(data.shape[0]):
            if isinstance(data[j, i], (float, int)) and not np.isnan(data[j,i]):
                if 0==1:
                    print('a')
            else:
                print(j,i)
                print(data[j,i])
                
                


