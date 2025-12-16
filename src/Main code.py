run=False
testing=False
tuning=True

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import rdFingerprintGenerator

import peptidy as pep

from peptidy import descriptors
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

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

    if colom_1[0]==0:
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

    def topological_fingerprints(self):
        """This function gets the topological fingerprints of an molecule
        
        input: self 
        
        output: an array"""
        topfingergen = AllChem.GetRDKitFPGenerator(fpSize=2048)
        topfinger = topfingergen.GetFingerprint(self.molecule)
        array = np.zeros((topfinger.GetNumBits(),), dtype=int)
        DataStructs.ConvertToNumpyArray(topfinger, array)
        return array
    
    def morgan_fingerprint(self,radius=2,nBits=1024):
        """Returns a Morgan fingerprint as a NumPy array.

        input: self for self.molecule, radius is an integer and defines the radius what it uses of the molecule
        nBits is an integer and defines length of the fingerprint vector

        output an array
        """
        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    
        # Generate the fingerprint (bit vector)
        fingerprint = morgan_generator.GetFingerprint(self.molecule)
    
        # Convert to numpy array
        array = np.zeros((nBits,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fingerprint, array)
    
        return array
    
    def macckeys(self):
        """This function gets the macckeys of an molecule
        
        input: self 
        
        output: an array"""
        macckeys = MACCSkeys.GenMACCSKeys(self.molecule)
        array = np.zeros((macckeys.GetNumBits(),), dtype=int)
        DataStructs.ConvertToNumpyArray(macckeys, array)
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

def train_validation_split(X_train, y_train, percentage):
    """This function splits the data randomly into training data set and a validation
    data set. These training and validation set are returned as a tuple of 2 tuples
    as (X_training,y_training),(X_validation, y_validation). It splits the the data in
    two with the percentage to determine how big training data set is. Percentage is
    a float between 1 and 0."""

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
    """CHecks if there are values in a matrix an random forest can crash on. This is an function for ourselves to control this if we get an error
    
    input matrix
    
    There is no output"""
    print(X)
    print('a')
    print("Heeft NaN:", np.isnan(X).any())
    print("Heeft +inf:", np.isinf(X).any())
    print("Heeft -inf:", np.isneginf(X).any())
    print("Max waarde:", np.nanmax(X))
    print("Min waarde:", np.nanmin(X))

def clipping_outliers(matrix,percentile_low=5,percentile_high=95):
    """This function changes outliers to the highest possible not outlier value, percentile_low, the smallest percentile,percentile_high, highest percentile both must be integers
    
    input: matrix (a colom is a feature)
    
    output: matrix same format"""
    new_array_list=[]
    for i in range (matrix.shape[1]):
        array=matrix[:,i]
        lowest_percentile=np.percentile(array,percentile_low)
        highest_percentile=np.percentile(array, percentile_high)
        output_array=np.clip(array,a_min=lowest_percentile,a_max=highest_percentile)
        new_array_list.append(output_array)
    
    matrix_output=np.column_stack(new_array_list)
    return matrix_output


def set_scaling(X):
    """makes the scaler, from given data set X. the scaler used
    is a minmax scaler. it returns a object with a fixed scaler"""
    scaler=sklearn.preprocessing.MinMaxScaler()
    return scaler.fit(X)

def data_scaling(scaler, X):
    """transforms data from fixed scalar. input is the fixed scaler
    and the data that need to be scaled. the output is th scaled data"""
    return scaler.transform(X)

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

def train_model(X,y,n_estimators=100,  criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
                oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None):
    """Trains the model
      
    Input:X is the matrix with features and samples, y is the affinityscore in an array which is coupled with the sample
    n_estimators is how many trees you want to use, this needs to be an integer, Max_depth is the maximum dept of the tree this is 
    an integer or None, min_samples_split this is an integer or float with how many samples are needed per split. Min_samples_leaf are the samples you need
    for a leaf node also an integer or fload, max_features how many features are used to make a tree can be sqrt, log2, none, bootstrap is True or False, Random_state is integer or false,
    Max_samples is an integer, float or None.
    
    Output: A random forest model  """
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
    random_forest=RandomForestRegressor(n_estimators=n_estimators,  criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                                         min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, 
                                                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, 
                                                        oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, 
                                                        ccp_alpha=ccp_alpha, max_samples=max_samples, monotonic_cst=monotonic_cst)
    random_forest.fit(X,y)
    return random_forest 


def combining_all_features(datafile, features=True, topological=True, morgan=True, macckeys=True):
    """This functions makes an matrix with the descriptors from the ligands and proteins in the file
    
    Input: csv-file with a format of the trainingset (trainset: first colom:SMILES, second colom:UNIProt_ID, third colom: affinity) or testset(first colom:numbers, second colom:SMILES, third colom:UNIProt_ID)
    
    Output: matrix (n_samples*n_features) and affinity (np.array of length n_samples)
    """
    SMILES,UNIProt_ID,affinity=data_to_SMILES_UNIProt_ID(datafile)
    uniprot_dict=extract_sequence("data/protein_info.csv")
    for i in range(len(SMILES)):
        ligand=small_molecule(SMILES[i])
        if features==True:
            ligand_features=ligand.rdkit_descriptor()
        if topological==True:
            ligand_topological=ligand.topological_fingerprints()
        if morgan==True:
            ligand_morgan=ligand.morgan_fingerprint()
        if macckeys==True:
            ligand_macckeys=ligand.macckeys()

        peptide=protein(UNIProt_ID[i], uniprot_dict)
        peptide_features_list=peptide.extract_features(peptide.uniprot2sequence())
        peptide_features=np.array(peptide_features_list)
        if features==True:
            if topological==True:
                if morgan==True:
                    if macckeys==True:
                        all_features=np.concatenate((ligand_features,ligand_topological,ligand_morgan,ligand_macckeys,peptide_features))
                    else:
                        all_features=np.concatenate((ligand_features,ligand_topological,ligand_morgan,peptide_features))
                else:
                    if macckeys==True:
                        all_features=np.concatenate((ligand_features,ligand_topological,ligand_macckeys,peptide_features))
                    else:
                        all_features=np.concatenate((ligand_features,ligand_topological,peptide_features))
            else:
                if morgan==True:
                    if macckeys==True:
                        all_features=np.concatenate((ligand_features,ligand_morgan,ligand_macckeys,peptide_features))
                    else:
                        all_features=np.concatenate((ligand_features,ligand_morgan,peptide_features))
                else:
                    if macckeys==True:
                        all_features=np.concatenate((ligand_features,ligand_macckeys,peptide_features))
                    else:
                        all_features=np.concatenate((ligand_features, peptide_features))
        else:
            if topological==True:
                if morgan==True:
                    if macckeys==True:
                        all_features=np.concatenate((ligand_topological,ligand_morgan,ligand_macckeys,peptide_features))
                    else:
                        all_features=np.concatenate((ligand_topological,ligand_morgan,peptide_features))
                else:
                    if macckeys==True:
                        all_features=np.concatenate((ligand_topological,ligand_macckeys,peptide_features))
                    else:
                        all_features=np.concatenate((ligand_topological,peptide_features))
            else:
                if morgan==True:
                    if macckeys==True:
                        all_features=np.concatenate((ligand_morgan,ligand_macckeys,peptide_features))
                    else:
                        all_features=np.concatenate((ligand_morgan,peptide_features))
                else:
                    if macckeys==True:
                        all_features=np.concatenate((ligand_macckeys,peptide_features))
                    else:
                        raise RuntimeError("at least one must be true")
                        
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


#Code voor Iris om te testen welke data source het beste is
def data_sources_training():
    X,y = combining_all_features('data/train.csv')
    train_set, validation_set = train_validation_split(X, y, 0.8)      #splits 20% of the data to the validation set, which is reserved for evaluation of the final model
    X_train_raw, y_train = train_set
    data_sources_dict = make_data_sources_dict(X_train_raw)
    best_data_source(data_sources_dict, y_train)
    return


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
        estimator = RandomForestRegressor(n_estimators=100,  criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
                oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
        mean_cv_score = cross_val_score(estimator, current_X_train, y_train, cv=5).mean()
        print(f'For the data source {current_data_source}, the mean cv score is {mean_cv_score}')
        if mean_cv_score > highest_cv_score:        
            highest_cv_score = mean_cv_score
            best_data_source = current_data_source          #keeps track of the best data source thus far
    print(f'The best data source is {best_data_source}')


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


def make_pca_plots(pca_scores):
    """makes three PCA-plots: first vs second PC, first vs third PC, and second vs third PC. 
    Input parameter: pca_scores (np.array): the data transformed onto the new PCA feature space.
    """
    fig, (ax1,ax2,ax3) = plt.subplots(3)
    fig.suptitle('Principal component plots on cleaned and scaled training data')
    ax1.scatter(pca_scores[:,0],pca_scores[:,1])
    ax1.set(xlabel='First PC explained variance',ylabel='Second PC explained variance')
    ax2.scatter(pca_scores[:,0],pca_scores[:,2])
    ax2.set(xlabel='First PC explained variance',ylabel='Third PC explained variance')
    ax3.scatter(pca_scores[:,1],pca_scores[:,2])
    ax3.set(xlabel='Second PC explained variance',ylabel='Third PC explained variance')
    plt.show()

def hyperparams_cv(X,y,param_grids, n_iter=100, cv_fold=5, search_type='randomized'):
    """Tunes the hyperparameters for the RF model using randomised search.
    Input:  
    X (np.array): array of size (n_samples * n_features)
    y (np.array): array of size (n_samples,)
    param_grids (dict): contains the parameters that will be tuned and their grid of values that will be tried
    n_iter (int): number of iterations the model will take, only relevant if search_type='randomized'
    cv_fold (int): determines the fold of the cross validation, i.e. how many different predictions will be made per parameter combination
    search_type ('randomized' or 'grid'): determines whether randomized or grid search will be performed.
    Returns a dictionary of the most optimal parameters found
    """
    model = RandomForestRegressor()
    if search_type=='grid':
        estimator = GridSearchCV(model, param_grids, n_jobs=-2, refit=True, cv=cv_fold)
    elif search_type=='randomized':
        estimator = RandomizedSearchCV(model, param_grids, n_jobs=-2, refit=True, cv=cv_fold, n_iter=n_iter, verbose=2)
    estimator.fit(X,y)
    best_estimator = estimator.best_estimator_
    best_params = estimator.best_params_
    return best_params


if tuning is True:
    starttime=time.time()
    print("started tuning")
    X,y=combining_all_features("data/train.csv",features=False,topological=True,morgan=True,macckeys=True)
    print("data is prepared")
    scaler=set_scaling(X)
    X_scaled=data_scaling(scaler,X)
    print("data is scaled")
    n_estimators_grid = range(100,301,20)
    max_depth_grid = range(3,16)
    min_samples_split_grid = range(2,11)
    min_samples_leaf_grid = range(1,6)
    max_features_grid = ['sqrt','log2',None]
    param_options = {'n_estimators':n_estimators_grid, 'max_depth':max_depth_grid, 'min_samples_split':min_samples_split_grid,
                     'min_samples_leaf':min_samples_leaf_grid, 'max_features':max_features_grid}
    print(hyperparams_cv(X,y,param_options,n_iter=200,cv_fold=3))
    total_time = time.time()-starttime
    print(f"this took {total_time} seconds, which is {total_time/60} minutes")

#This if statement is really useful if you want to work on other parts of the code                
if run is True:                
    starttime=time.time()
    print("started")
    bestscore=0
    bestfeatures=False
    besttopological=True
    bestmorgan=True
    bestmacckeys=True
    if testing==True:
        for f in range(0,2):
            for t in range(0,2):
                for mo in range(0,2):
                    for ma in range(0,2):
                        if f==1 and t==1 and mo==1 and ma==1:
                            pass
                        else:
                            if f==0:
                                features=True
                            else:
                                features=False
                            if t==0:
                                topological=True
                            else:
                                topological=False
                            if mo==0:
                                morgan=True
                            else:
                                morgan=False
                            if ma==0:
                                macckeys=True
                            else:
                                macckeys=False
                            X,y=combining_all_features("data/train.csv",features=features,topological=topological,morgan=morgan,macckeys=macckeys)
                            training,validation=train_validation_split(X,y,0.8)
                            X_training,y_training=training
                            X_validation,y_validation=validation
                            print("data is prepared")
                            scaler=set_scaling(X_training)
                            X_training_scaled=data_scaling(scaler,X_training)
                            X_validation_scaled=data_scaling(scaler,X_validation)
                            print("data is scaled")
                            model=train_model(X_training_scaled,y_training)
                            print("model is trained")
                            score=RF_error(model,X_validation_scaled,y_validation)
                            print("features="+str(features)+" and topological="+str(topological)+" and morgan="+str(morgan)+" and macckeys="+str(macckeys)+" -> score="+str(score))
                            if score>bestscore:
                                bestscore=score
                                bestfeatures=features
                                besttopological=topological
                                bestmorgan=morgan
                                bestmacckeys=macckeys
                            print("score is calculated for features="+str(features)+" and topological="+str(topological)+" and morgan="+str(morgan)+" and macckeys="+str(macckeys))
    X,y=combining_all_features("data/train.csv",features=bestfeatures,topological=besttopological,morgan=bestmorgan,macckeys=bestmacckeys)
    print("trainingset is prepared")
    scaler=set_scaling(X)
    X_scaled=data_scaling(scaler,X)
    print("trainingset is scaled")
    X_test,unknown_affinity=combining_all_features("data/test.csv",features=bestfeatures,topological=besttopological,morgan=bestmorgan,macckeys=bestmacckeys)
    print("testset is prepared")
    X_test_scaled=data_scaling(scaler,X_test)
    print("testset is scaled")
    model=train_model(X_scaled,y)
    print("model is trained")
    kaggle_submission(X_test_scaled,model,"docs/Kaggle_submission.csv")
    print("file is made with predictions")
    endtime=time.time()
    print("the model is trained en data is predicted")
    print("this took " + str(endtime-starttime) + "seconds")

