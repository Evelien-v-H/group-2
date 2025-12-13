run=False

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import GraphDescriptors

import peptidy as pep

from peptidy import descriptors
import sklearn
from sklearn.ensemble import RandomForestRegressor

import time

#Openen data:
def open_data(datafile):
    """The files with data need to be imported.
    
    Input: a csv datafile
    
    Output: a dataFrame"""
    df=pd.read_csv(datafile)
    return df


def data_to_SMILES_UNIProt_ID(datafile):
    """This function splits the dataset in a SMILES array and UNIProt_ID. If the datafile is an trainset than also an affinityscore 
        
        input: a csv file like the given training or testset with 3 coloms. (trainset: first colom:SMILES, second colom:UNIProt_ID, third colom: affinity)
        (testset: first colom:numbers, second colom:SMILES, third colom:UNIProt_ID)
    
        Output: one array with the SMILES, an array with the UNIProt_IDs and an array with the affinityscore, if it is the testset the
        affinityscore is returned with 'unknown affinity'"""
    df=open_data(datafile)
    colom_1=df.iloc[:,0].to_numpy()
    colom_2=df.iloc[:,1].to_numpy()
    colom_3=df.iloc[:,2].to_numpy()

    if colom_1[0]=="0":
        SMILES=colom_2
        UNIProt_ID=colom_3
        affinity='unknown affinity'
    
    else:
        SMILES=colom_1
        UNIProt_ID=colom_2
        affinity=colom_3

    return SMILES,UNIProt_ID,affinity


def extract_sequence(document):
    """input is a doc but the document earlier made should be in the format of:
    first the uniprot-id, second the proteinacronym and last the one
    letter code sequence of the protein, split by comma's. the first
    line is to tell, which column is which. the output is a dictionary,
    with as keys the uniprotid and as value the one letter code 
    sequence of the protein. (For this project it is the protein_info.csv)"""
    file=open(document) #document needs to be in the right format
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

class small_molecule:
    def __init__(self,SMILES):
        self.SMILES=str(SMILES)
        self.molecule=Chem.MolFromSmiles(str(SMILES))




    def rdkit_descriptor(self):
        """This function returns an array with all sorts of descriptors gotten from rdkit
        
        input: self
        
        output an array"""
        dictionary=Descriptors.CalcMolDescriptors(self.molecule, missingVal=None, silent=True)
        descriptor_list=list((dictionary.values()))
        descriptor_list.pop(42)
        
        #With testing the code we found out that IPC the descriptor that is now deleted gave problems 
        #if it was implemented this way, because it returned to big numbers. This is an known phenonemon and an easy work around as 
        #implemented here

        IPC=GraphDescriptors.Ipc(self.molecule,avg=True)
        descriptor_list.append(IPC)
        array=np.array(descriptor_list)
        
        return array


    
class protein:
    def __init__(self, uniprot_id, dict):
        self.uniprot_id = uniprot_id
        self.dictionary=dict

    def uniprot2sequence(self):
        """no input, returns a string with the protein one letter code sequence"""
        uniprot_dict=self.dictionary
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
    return scaler.fit(X)

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
    
def combining_all_features(datafile):
    """This functions makes an matrix with the descriptors from the ligands and proteins in the file
    
    Input: csv-file with a format of the trainingset (trainset: first colom:SMILES, second colom:UNIProt_ID, third colom: affinity) or testset(first colom:numbers, second colom:SMILES, third colom:UNIProt_ID)
    
    Output: matrix (n_samples*n_features) and affinity (np.array of length n_samples)
    """
    SMILES,UNIProt_ID,affinity=data_to_SMILES_UNIProt_ID(datafile)
    uniprot_dict=extract_sequence("data/protein_info.csv")
    for i in range(len(SMILES)):
        ligand=small_molecule(SMILES[i])
        ligand_features=ligand.rdkit_descriptor()

        peptide=protein(UNIProt_ID[i], uniprot_dict)
        peptide_features_list=peptide.extract_features(peptide.uniprot2sequence())
        peptide_features=np.array(peptide_features_list)
        all_features=np.concatenate((ligand_features, peptide_features))
        if i==0:
            matrix=all_features
        
        else:
            matrix=np.vstack((matrix,all_features))

    return matrix,affinity


def fit_PCA(X, n_components=None):
    """performs a PCA on the data in X (np.array) and 
    returns the features transformed onto the principal component feature space as X_scores (np.array of shape (n_samples, n_components))
    Input parameter n_components has default value None, meaning it keeps all principal components by default"""
    pca = sklearn.decomposition.PCA(n_components=n_components)
    X_scores = pca.fit_transform(X)
    variance_per_pc = pca.explained_variance_ratio_
    return X_scores, variance_per_pc

#Onderstaande 2 functies zijn voor ons eigen gebruik, voor als we gaan testen welke data source het beste is (Iris, vrijdagavond)
def make_data_sources_dict(X_train_raw):
    """applies cleaning, scaling, and pca where relevant.
    returns the dictionary that can be used for the test_data_source function. 
    This dictionary includes all data that will be tried for selecting the best input data:
        - scaled
        - cleaned (= outliers removed)
        - cleaned + scaled
        - scaled and transformed into pca feature space with enough features to explain 60% of variance
        - scaled and transformed, 80%
        - scaled and transformed, 95%
        - the previous three but with the cleaning step"""
    
    scaler = set_scaling(X_train_raw)
    X_train_scaled = data_scaling(scaler, X_train_raw)
    X_train_pc_scores, variance_per_pc = fit_PCA(X_train_scaled)
    X_train_pca66 = select_principal_components(X_train_pc_scores, variance_per_pc, 0.66)
    X_train_pca80 = select_principal_components(X_train_pc_scores, variance_per_pc, 0.80)
    X_train_pca95 = select_principal_components(X_train_pc_scores, variance_per_pc, 0.95)
    X_train_cleaned = data_cleaning(X_train_raw)
    X_train_cleaned_scaled = data_scaling(scaler, X_train_cleaned)
    X_train_cleaned_pc_scores, variance_per_pc_cleaned = fit_PCA(X_train_cleaned_scaled)
    X_train_cleaned_pca66 = select_principal_components(X_train_cleaned_pc_scores, variance_per_pc_cleaned, 0.66)
    X_train_cleaned_pca80 = select_principal_components(X_train_cleaned_pc_scores, variance_per_pc_cleaned, 0.80)
    X_train_cleaned_pca95 = select_principal_components(X_train_cleaned_pc_scores, variance_per_pc_cleaned, 0.95)

    data_sources_dict = {'Scaled':X_train_scaled, 'Cleaned':X_train_cleaned, 'Cleaned+scaled':X_train_cleaned_scaled, 
                         'Scaled+pca66':X_train_pca66, 'Scaled+pca80':X_train_pca80, 'Scaled+pca95':X_train_pca95,
                         'Cleaned+scaled+pca66':X_train_cleaned_pca66, 'Cleaned+scaled+pca80':X_train_cleaned_pca80, 'Cleaned+scaled+pca95':X_train_cleaned_pca95}
    return data_sources_dict

def best_data_source(data_sources_dict, y_train):
    """tries multiple data sources specified in data_sources_dict to determine the best one using cross-validation"""
    highest_cv_score = 0
    for current_data_source, current_X_train in data_sources_dict.items():                            #loops over the different data sources in the dictionary, data_source is the index of the current iteration
        clf = sklearn.ensemble.RandomForestRegressor(n_estimators=100,  criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
                oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
        mean_cv_score = sklearn.model_selection.cross_val_score(clf, current_X_train, y_train, cv=5).mean()
        print(f'For the data source {current_data_source}, the mean cv score is {mean_cv_score}')
        if mean_cv_score > highest_cv_score:        
            highest_cv_score = mean_cv_score
            best_data_source = current_data_source          #keeps track of the best data source thus far
    print(f'The best data source is {best_data_source}')


#Code voor Iris om te testen welke data source het beste is
def data_sources_training():
    X,y = combining_all_features('data/train.csv')
    train_set, validation_set = splittingdata(X, y, 0.2)      #splits 20% of the data to the validation set, which is reserved for evaluation of the final model
    X_train_raw, y_train = train_set
    data_sources_dict = make_data_sources_dict(X_train_raw)
    best_data_source(data_sources_dict, y_train)
    return

def select_principal_components(X_pca_scores, variance_explained, goal_cumulative_variance):
    """from the input array X_pca_scores, creates a new array relevant_principle_components, which is a subset
    of the input array that includes only the relevant PCs to reach the goal_cumulative_variance. 
    variance_explained is a np.array of shape (n_principal_components,) 
    returned by the function fit_PCA that contains the portion of variance explained by each principal component."""
    cumulative_variance = 0
    pc = 0
    while cumulative_variance < goal_cumulative_variance:
        cumulative_variance += variance_explained[pc]
        pc += 1
    relevant_principal_components = X_pca_scores[:, :pc]
    return relevant_principal_components


def kaggle_submission(X_test,model,filename):
    affinity_array=RF_predict(model, X_test)
    f=open(filename,'w')
    print(filename+" is made")
    f.write("ID,affinity_score")
    b=0
    for a in affinity_array:
        f.write("\n"+str(b)+","+ str(a))
        b+=1
    f.close()
    return


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
                
if run is True:                
    starttime=time.time()
    print("started")
    X,y=combining_all_features("data/train.csv")
    X_test,unknown_affinity=combining_all_features("data/test.csv")
    print("data is prepared")
    scaler=set_scaling(X)
    X_scaled=data_scaling(scaler,X)
    X_test_scaled=data_scaling(scaler,X_test)
    print("data is scaled")
    model=train_model(X_scaled,y)
    print("model is trained")
    kaggle_submission(X_test_scaled,model,"docs/Kaggle_submission.csv")
    print("file is made with predictions")
    endtime=time.time()
    print("the model is trained en data is predicted")
    print("this took " + str(endtime-starttime) + "seconds")

def is_number(val):
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def data_cleaning(data):
    """Cleans the data, removes coloms without floats or strings and replaces empty values or strings in features and gives an error 
    if een float or int is to big for np.float32
    
    Input data matrix (n*m)
    
    Output an matrix, size is (n*unknown) unknown is dependend on the useless features because string or none information """
    irrelevant_colums=[]
    for i in range(data.shape[1]):
        already_errorvalue=False
        for j in range(data.shape[0]):
            if is_number(data[j,i]) is True:
                if float(data[j,i])>=np.finfo(np.float32).max:
                    raise ValueError("You're value is to big for the float32 of the random forest. Solve this manual")
                else:
                    data[j,i]=float(data[j,i])
            
            else:
                if already_errorvalue is False:
                    values_colom=[]
                    for k in range(data.shape[0]):
                        if is_number(data[k,i]) is True:
                            values_colom.append(float(data[k,i]))
                    
                    if len(values_colom)!=0:
                        mean_value_colom=np.mean(values_colom)
                        data[j,i]=float(mean_value_colom)
                    
                    else:
                        irrelevant_colums.append(i)
                    already_errorvalue=True
                    
            
                else:
                    if i not in irrelevant_colums:
                        data[j,i]=float(mean_value_colom)

    if len(irrelevant_colums)>0:
        irrelevant_colums.reverse()
        for n in irrelevant_colums:
            print('a feature is deleted colom:',n)
            data = np.delete(data, n, axis=1)

    return data

def check_matrix(X):
    print(X)
    print('a')
    print("Heeft NaN:", np.isnan(X).any())
    print("Heeft +inf:", np.isinf(X).any())
    print("Heeft -inf:", np.isneginf(X).any())
    print("Max waarde:", np.nanmax(X))
    print("Min waarde:", np.nanmin(X))

def make_pca_plots(pca_scores):
    """makes three PCA-plots: first vs second PC, first vs third PC, and second vs third PC. 
    Input parameter: pca_scores (np.array): the data transformed onto the new PCA feature space."""
    fig, (ax1,ax2,ax3) = plt.subplots(3)
    fig.suptitle('Principal component plots on cleaned and scaled training data')
    ax1.scatter(pca_scores[:,0],pca_scores[:,1])
    ax1.set(xlabel='First PC explained variance',ylabel='Second PC explained variance')
    ax2.scatter(pca_scores[:,0],pca_scores[:,2])
    ax2.set(xlabel='First PC explained variance',ylabel='Third PC explained variance')
    ax3.scatter(pca_scores[:,1],pca_scores[:,2])
    ax3.set(xlabel='Second PC explained variance',ylabel='Third PC explained variance')
    plt.show()


