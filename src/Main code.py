run=False
kaggle=True
tuning=False

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

from peptidy import *
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
        onehot=pep.one_hot_encoding(sequence,822) #longest protein is 822 amino acids
        return onehot
    
    def extract_global_descriptors(self, sequence):
        """extracts the protein features from the amino acid sequence using peptidy. 
        Returns a list of all numerical features from peptidy of this protein"""    
        peptidy_features_dict = pep.descriptors.compute_descriptors(sequence, descriptor_names=None, pH=7)
        peptidy_global_features_list = list(peptidy_features_dict.values())
        return peptidy_global_features_list
    
    def extract_all_local_descriptors(self, sequence):
        all_aa_descr = pep.encoding.aminoacid_descriptor_encoding(sequence, descriptor_names=None)
        return np.array(all_aa_descr)  #shape: (n_residues, n_descriptors)
    
    def compute_window_based_features(self, sequence, all_residue_descr):
        """
        First computes mean and variance of each residue descriptor within a certain window. 
        Then, computes mean, sum, variance, and max of these window statistics. 
        This results in 8 new features per residue descriptor per window size.
        With the three window sizes for short-range, medium-range, and long-range interactions, 
        this results in 24 features per residue descriptor. 
          """
        n_residues, n_descr = np.shape(all_residue_descr)
        aggregated_window_descr = []
        for window_size in [4,8,15]:            #three different window sizes for short, medium, and long-range interactions
            for descriptor in range(n_descr):
                window_statistics = []
                for window_start in range(0, len(sequence), window_size):
                    window_stop = window_start + window_size
                    mean = np.mean(all_residue_descr[window_start:window_stop, descriptor])
                    variance = np.var(all_residue_descr[window_start:window_stop, descriptor])
                    window_statistics.append([mean, variance]) 
                window_statistics = np.array(window_statistics)             #shape: (n_residues/window_size , 2)

                for window_statistic in range(np.shape(window_statistics)[1]):
                    mean = np.mean(window_statistics[:, window_statistic])      #calculates the mean over all windows of each window statistic
                    sum = np.sum(window_statistics[:, window_statistic])
                    variance = np.var(window_statistics[:, window_statistic])
                    max_val = max(window_statistics[:, window_statistic])
                    aggregated_window_descr.extend([mean, sum, variance, max_val])
        
        return aggregated_window_descr      #a long list of all window-based protein descriptors, length = 24*n_descr

    def compute_autocorrelation_features(self, all_residue_descr):
        """computes the autocorrelation for three different lags, all 
        having different biological relevance because they capture different ranges of interaction."""
        n_residues, n_descr = np.shape(all_residue_descr)
        autocorrelation_features = []
        for descr in range(n_descr):
            current_descr_values = all_residue_descr[:, descr]
            descr_scaled = current_descr_values - np.mean(current_descr_values)
            correlation = np.correlate(descr_scaled, descr_scaled, mode='full')
            for lag in [1,4,10]:
                autocorrelation_features.append(correlation[lag])
        return autocorrelation_features                 #a list of all autocorrelation_based features, length is 3*n_descr


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

def data_cleaning_train(data):
    """Cleans the data, removes coloms without floats or strings and replaces empty values or strings in features and gives an error 
    if een float or int is to big for np.float32 and it returns information about this proces
    
    Input data matrix (n*m)
    
    Output an matrix, size is (n*unknown) unknown is dependend on the useless features because string or none information, an list with information about the mean value of coloms
    and an list with irrelevant features, this information is needed for the test set """
    irrelevant_colums=[]
    mean_value_coloms=[]
    for i in range(data.shape[1]):
        values_colom=[]
        for j in range(data.shape[0]):
            if is_number(data[j,i]) is True:
                values_colom.append(float(data[j,i]))

        if len(values_colom)!=0:
            mean_value=np.mean(values_colom)
            mean_value_coloms.append(mean_value)
                    
        else:
            irrelevant_colums.append(i)

        for k in range(data.shape[0]):
            if i not in irrelevant_colums:
                if is_number(data[k,i]) is True:
                    if float(data[k,i])>=np.finfo(np.float32).max:
                        raise ValueError("Youre value is to big for the float32 of the random forest. Solve this manual")
                    else:
                        data[k,i]=float(data[k,i])
            
                else:
                    data[k,i]=mean_value

    if len(irrelevant_colums)>0:
        irrelevant_colums.reverse()
        for n in irrelevant_colums:
            print('a feature is deleted colom:',n)
            data = np.delete(data, n, axis=1)

    return data, mean_value_coloms, irrelevant_colums

def data_cleaning_test(data,mean_value_coloms, irrelevant_colums):
    """This code cleans the test data with information from the cleaning of the trainingset
    
    input: data in matrix, mean_value_coloms in a list and irrelevant_colomus in an list. This list needs to be high values to lower values
    
    output: data in a matrix"""
    if len(irrelevant_colums)>0:
         for n in irrelevant_colums:
            print('a feature is deleted colom:',n)
            data = np.delete(data, n, axis=1)
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if is_number(data[j,i]) is True:
                if float(data[j,i])>=np.finfo(np.float32).max:
                    raise ValueError("Youre value is to big for the float32 of the random forest. Solve this manual")
                else:
                    data[j,i]=float(data[j,i])
            else:
                data[j,i]=mean_value_coloms[i]
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

def clipping_outliers_train(matrix,percentile_low=5,percentile_high=95):
    """This function changes outliers to the highest possible not outlier value, percentile_low, the smallest percentile,percentile_high, highest percentile both must be integers
    
    input: matrix (a colom is a feature)
    
    output: matrix same format, list with lowest and highest percentile in an tuple for every colom, this is needed for the test clipping outlier"""
    new_array_list=[]
    percentile_list=[]
    for i in range (matrix.shape[1]):
        array=matrix[:,i]
        lowest_percentile=np.percentile(array,percentile_low)
        highest_percentile=np.percentile(array, percentile_high)
        output_array=np.clip(array,a_min=lowest_percentile,a_max=highest_percentile)
        new_array_list.append(output_array)
        percentile_list.append((lowest_percentile,highest_percentile))
    
    matrix_output=np.column_stack(new_array_list)
    return matrix_output,percentile_list

def clipping_outliers_test(matrix, percentile_list):
    """"This function changes outliers to the highest possible not outlier value, percentile_list contains the info which the function uses for that purpose
    
    input: matrix, percentile_list (first lowest value, than highest value)
    
    output matrix"""
    new_array_list=[]
    for i in range (matrix.shape[1]):
        array=matrix[:,i]
        lowest_percentile,highest_percentile=percentile_list[i]
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
    
    random_forest=RandomForestRegressor(n_estimators=n_estimators,  criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split,
                                                         min_samples_leaf=min_samples_leaf,min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, 
                                                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap, 
                                                        oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, 
                                                        ccp_alpha=ccp_alpha, max_samples=max_samples, monotonic_cst=monotonic_cst)
    random_forest.fit(X,y)
    return random_forest 


def extract_all_features(datafile,ligandf=True,topologicalf=True,morganf=True,
                         macckeysf=True,peptidef=True,windowbasedf=True,autocorrelationf=True):
    """this function makes a dictionary with all the different datasets. 
    Input is a csv-file with a format of the trainingset and testset.
    Other inputs are for determining which features need to be extracted.
    Output is a dictionary with as key the name of the feature and as 
    value an array of values per sample"""

    SMILES,UNIProt_ID,affinity=data_to_SMILES_UNIProt_ID(datafile)
    uniprot_dict=extract_sequence("data/protein_info.csv")
    dictionary={}
    ligand_list=[]
    topological_list=[]
    morgan_list=[]
    macckeys_list=[]
    peptide_list=[]
    windowbased_list=[]
    autocorrelation_list=[]

    for i in range(len(SMILES)):
        ligand=small_molecule(SMILES[i])
        peptide=protein(UNIProt_ID[i], uniprot_dict)
        sequence=peptide.uniprot2sequence()
        all_residue_descr=peptide.extract_all_local_descriptors(sequence)
        if ligandf==True:
            ligand_features=ligand.rdkit_descriptor()
            ligand_list.append(ligand_features)
        if topologicalf==True:
            ligand_topological=ligand.topological_fingerprints()
            topological_list.append(ligand_topological)
        if morganf==True:
            ligand_morgan=ligand.morgan_fingerprint()
            morgan_list.append(ligand_morgan)
        if macckeysf==True:
            ligand_macckeys=ligand.macckeys()
            macckeys_list.append(ligand_macckeys)
        if peptidef==True:
            peptide_features_list=peptide.extract_global_descriptors(sequence)
            peptide_features=np.array(peptide_features_list)
            peptide_list.append(peptide_features)
        if windowbasedf==True:
            peptide_windowbased_list=peptide.compute_window_based_features(sequence,all_residue_descr)
            peptide_windowbased=np.array(peptide_windowbased_list)
            windowbased_list.append(peptide_windowbased)
        if autocorrelationf==True:
            peptide_autocorrelation_list=peptide.compute_autocorrelation_features(all_residue_descr)
            peptide_autocorrelation=np.array(peptide_autocorrelation_list)
            autocorrelation_list.append(peptide_autocorrelation)

    if ligandf==True:
        ligand_array=np.array(ligand_list)
        n_samples, lf_features=ligand_array.shape
        dictionary['ligandf']=ligand_array,lf_features
    if topologicalf==True:
        topological_array=np.array(topological_list)
        n_samples, tf_features=topological_array.shape
        dictionary['topologicalf']=topological_array,tf_features
    if morganf==True:
        morgan_array=np.array(morgan_list)
        n_samples,mo_features=morgan_array.shape
        dictionary['morganf']=morgan_array,mo_features
    if macckeysf==True:
        macckeys_array=np.array(macckeys_list)
        n_samples,ma_feautures=macckeys_array.shape
        dictionary['macckeysf']=macckeys_array,ma_feautures
    if peptidef==True:
        peptide_array=np.array(peptide_list)
        n_samples,pf_features=peptide_array.shape
        dictionary['peptidef']=peptide_array,pf_features
    if windowbasedf==True:
        windowbased_array=np.array(windowbased_list)
        n_samples,wb_features=windowbased_array.shape
        dictionary['windowbasedf']=windowbased_array,wb_features
    if autocorrelationf==True:
        autocorrelation_array=np.array(autocorrelation_list)
        n_samples,ac_features=autocorrelation_array.shape
        dictionary['autocorrelationf']=autocorrelation_array,ac_features
    
    return dictionary,affinity

def slicing_features(large_feature_array, n_features_list, bool_list):
    """
    Parameters:
        large_feature_array (np.array): array of shape (n_samples, total_n_features) with all 
            possible small molecule encodings and protein encodings, in the following order: ligand features, topological fingerprints,
            morgan fingerprints, macckeys, peptide features (for the protein as a whole), window-based features, and autocorrelation features.
        n_features_list (list): list of the number of features corresponding to each of the feature types described above, in the same order.
        bool_list (list): contains the booleans corresponding to each feature, in the same order.
    
    Returns:
        sliced_features (np.array): np.array of shape (n_samples, n_features) that consists of all features of which the boolean
            input parameter was set to True.
    
    """                               #keeps track of the encodings included in the output
    cumulative_n_features = np.cumsum(n_features_list)
    sliced_features=None

    for i in range(len(bool_list)):
        if bool_list[i]:
            start_index = cumulative_n_features[i]
            stop_index = start_index + n_features_list[i]
            array_to_be_added = large_feature_array[:,start_index:stop_index]
            if sliced_features is None:                       
                sliced_features = array_to_be_added         #it is impossible to concatenate something to an empty array
            else:
                sliced_features = np.concatenate((sliced_features, array_to_be_added), axis=1)

    return sliced_features

def create_tf_combinations(remaining, current):
    """returns list of lists of all possible combinations of True and False. Remaining (int) indicates the length of each list of booleans. 
    Current should always be [] for this algorithm to work."""
    if remaining == 0:
        return [current]
    with_true = create_tf_combinations(remaining - 1, current + [True])
    with_false = create_tf_combinations(remaining - 1, current + [False])
    return with_true + with_false

def verify_tf_combinations(tf_combinations):
    """ensures all true-false combination lists that are passed onto the rest of the code will have at least one small molecule encoding and one protein encoding
    set to True. Input: output from function create_tf_combinations."""
    valid_tf_combinations = []
    for list_of_bools in tf_combinations:
        if (list_of_bools[0] or list_of_bools[1] or list_of_bools[2] or list_of_bools[3]) and (list_of_bools[4] or list_of_bools[5] or list_of_bools[6]):
            valid_tf_combinations.append(list_of_bools)
    return valid_tf_combinations

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
def data_sources_training(data_sources_dict,n_features_list,affinity, encoding_booleans):
    splitted_data_dict={}
    for current_data_source, current_X_train in data_sources_dict.items():
        splitted_X=slicing_features(current_X_train,n_features_list,encoding_booleans)
        splitted_data_dict[current_data_source]=splitted_X
    
    highest_cv_score,best_datasource=best_data_source(splitted_data_dict, affinity)
    return highest_cv_score,best_datasource

def make_data_sources_dict(X_train_raw,PCA=True):
    """applies cleaning, scaling, and pca where relevant.
    returns the dictionary that can be used for the test_data_source function. 
    This dictionary includes all data that will be tried for selecting the best input data:
        - scaled
        - cleaned (=clipping outliers)
        - cleaned + scaled
        - scaled and transformed into pca feature space with enough features to explain 60% of variance
        - scaled and transformed, 80%
        - scaled and transformed, 95%
        - the previous three but with the cleaning step"""
    
    X_train, mean_value_coloms, irrelevant_colums=data_cleaning_train(X_train_raw)

    scaler = set_scaling(X_train)
    X_train_scaled = data_scaling(scaler, X_train)
    if PCA==True:
        X_train_pc_scores, variance_per_pc = fit_PCA(X_train_scaled)
        X_train_pca66 = select_principal_components(X_train_pc_scores, variance_per_pc, 0.66)
        X_train_pca80 = select_principal_components(X_train_pc_scores, variance_per_pc, 0.80)
        X_train_pca95 = select_principal_components(X_train_pc_scores, variance_per_pc, 0.95)
    X_train_cleaned, percetile_list = clipping_outliers_train(X_train)
    X_train_cleaned_scaled = data_scaling(scaler, X_train_cleaned)
    if PCA==True:
        X_train_cleaned_pc_scores, variance_per_pc_cleaned = fit_PCA(X_train_cleaned_scaled)
        X_train_cleaned_pca66 = select_principal_components(X_train_cleaned_pc_scores, variance_per_pc_cleaned, 0.66)
        X_train_cleaned_pca80 = select_principal_components(X_train_cleaned_pc_scores, variance_per_pc_cleaned, 0.80)
        X_train_cleaned_pca95 = select_principal_components(X_train_cleaned_pc_scores, variance_per_pc_cleaned, 0.95)
    

    data_sources_dict = {'Scaled':X_train_scaled, 'Cleaned':X_train_cleaned, 'Cleaned+scaled':X_train_cleaned_scaled}
    if PCA==True:
        data_sources_dict['Scaled+pca66']=X_train_pca66
        data_sources_dict['Scaled+pca80']=X_train_pca80
        data_sources_dict['Scaled+pca95']=X_train_pca95
        data_sources_dict['Cleaned+scaled+pca66']=X_train_cleaned_pca66
        data_sources_dict['Cleaned+scaled+pca80']=X_train_cleaned_pca80
        data_sources_dict['Cleaned+scaled+pca95']=X_train_cleaned_pca95
    
    return data_sources_dict


def best_data_source(data_sources_dict, y_train):
    """tries multiple data sources specified in data_sources_dict to determine the best one using cross-validation"""
    highest_cv_score = 0
    for current_data_source, current_X_train in data_sources_dict.items():                            #loops over the different data sources in the dictionary, data_source is the index of the current iteration
        estimator = RandomForestRegressor(n_estimators=100,  criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, 
                oob_score=False, n_jobs=-2, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None)
        mean_cv_score = cross_val_score(estimator, current_X_train, y_train, cv=3).mean()
        print(f'For the data source {current_data_source}, the mean cv score is {mean_cv_score}')
        if mean_cv_score > highest_cv_score:        
            highest_cv_score = mean_cv_score
            best_datasource = current_data_source          #keeps track of the best data source thus far
    print(f'The best data source is {best_datasource} with score={highest_cv_score}')
    return highest_cv_score, best_datasource


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
    data_dictionary,affinity=extract_all_features("data/train.csv")
    lf_array,lf_features=data_dictionary['ligandf']
    tf_array,tf_features=data_dictionary['topologicalf']
    mo_array,mo_features=data_dictionary['morganf']
    ma_array,ma_features=data_dictionary['macckeysf']
    pf_array,pf_features=data_dictionary['peptidef']
    wb_array,wb_features=data_dictionary['windowbasedf']
    ac_array,ac_features=data_dictionary['autocorrelationf']
    all_features=np.concatenate([lf_array,tf_array,mo_array,ma_array,pf_array,wb_array,ac_array],axis=1)
    n_features_list=[lf_features,tf_features,mo_features,ma_features,pf_features,wb_features,ac_features]
    order_of_encodings = ['ligandf', 'topological', 'morgan', 'macckeys', 'peptidef', 'windowbased', 'autocorrelation']
    encoding_bools = []         #Here, you can insert booleans corresponding to every encoding. This is in the order of the list in the previous line.
    
    X = slicing_features(all_features, n_features_list, encoding_bools)
    y = affinity

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

    data_dictionary,affinity=extract_all_features("data/train.csv")
    lf_array,lf_features=data_dictionary['ligandf']
    tf_array,tf_features=data_dictionary['topologicalf']
    mo_array,mo_features=data_dictionary['morganf']
    ma_array,ma_features=data_dictionary['macckeysf']
    pf_array,pf_features=data_dictionary['peptidef']
    wb_array,wb_features=data_dictionary['windowbasedf']
    ac_array,ac_features=data_dictionary['autocorrelationf']

    all_features=np.concatenate([lf_array,tf_array,mo_array,ma_array,pf_array,wb_array,ac_array],axis=1)
    n_features_list=[lf_features,tf_features,mo_features,ma_features,pf_features,wb_features,ac_features]
    order_of_encodings = ['ligandf', 'topological', 'morgan', 'macckeys', 'peptidef', 'windowbased', 'autocorrelation']

    data_sources_dict=make_data_sources_dict(all_features,PCA=False)
    true_false_combinations = create_tf_combinations(len(n_features_list),[])
    valid_tf_combinations = verify_tf_combinations(true_false_combinations)

    for encoding_bools in valid_tf_combinations:
        included_encodings = []
        for i in range(len(encoding_bools)):
            if encoding_bools[i]:
                included_encodings.append(order_of_encodings[i])
        print(f'This iterations uses the features from: {included_encodings}')
        score, best_datasource=data_sources_training(data_sources_dict,n_features_list,affinity, encoding_booleans=encoding_bools)
        print(f'The score is {score}')                            
        if score>bestscore:
            bestscore = score
            bestbools = encoding_bools
            bestligandf = bestbools[0]
            besttopological = bestbools[1]
            bestmorgan = bestbools[2]
            bestmacckeys = bestbools[3]
            bestpeptidef = bestbools[4]
            bestwindowbased = bestbools[5]
            bestautocorrelation = bestbools[6]
            bestdatasource = best_datasource

        print('')

    print("training took "+str((time.time()-starttime)/3600)+" hours")
    print("")
    print("best result is: "+str(bestscore))
    print("and is achieved with the following adjustments:")
    print("ligandf="+str(bestligandf)+" and topological="+str(besttopological)+" and morgan="+str(bestmorgan)+" and macckeys="+str(bestmacckeys))
    print("peptidef="+str(bestpeptidef)+" and windowbased="+str(bestwindowbased)+" and autocorrelation="+str(bestautocorrelation))
    print("datasource="+str(bestdatasource))

    


if kaggle==True:
    starttime=time.time()
    data_dictionary_train, y = extract_all_features("data/train.csv")
    data_dictionary_test, unknown_affinity  = extract_all_features("data/test.csv")

    order_of_encodings = ['ligandf', 'topological', 'morgan', 'macckeys', 'peptidef', 'windowbased', 'autocorrelation']
    encoding_bools = [True,True,False,False,True,False,False] #Here, you can insert booleans corresponding to every encoding. This is in the order of the list in the previous line.
    scaling=False
    cleaning=True

    for data_dictionary in (data_dictionary_train, data_dictionary_test):
        lf_array,lf_features=data_dictionary['ligandf']
        tf_array,tf_features=data_dictionary['topologicalf']
        mo_array,mo_features=data_dictionary['morganf']
        ma_array,ma_features=data_dictionary['macckeysf']
        pf_array,pf_features=data_dictionary['peptidef']
        wb_array,wb_features=data_dictionary['windowbasedf']
        ac_array,ac_features=data_dictionary['autocorrelationf']

        n_features_list=[lf_features,tf_features,mo_features,ma_features,pf_features,wb_features,ac_features]
        
        if data_dictionary is data_dictionary_train:
            all_features_train=np.concatenate([lf_array,tf_array,mo_array,ma_array,pf_array,wb_array,ac_array],axis=1)
            X = slicing_features(all_features_train, n_features_list, encoding_bools)
        else:
            all_features_test=np.concatenate([lf_array,tf_array,mo_array,ma_array,pf_array,wb_array,ac_array],axis=1)
            X_test = slicing_features(all_features_test, n_features_list, encoding_bools)
    
    print("trainingset is prepared")
    print("testset is prepared")
    if scaling is True:
        scaler=set_scaling(X)
        X_scaled=data_scaling(scaler,X)
        print("trainingset is scaled")
        X_test_scaled=data_scaling(scaler,X_test)
        print("testset is scaled")
        model=train_model(X_scaled,y, n_estimators=350,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=21)
        print("model is trained")
        kaggle_submission(X_test_scaled,model,"docs/Kaggle_submission.csv")

    if cleaning is True:
        X_cleaned,clean=clipping_outliers_train(X)
        X_test_cleaned=clipping_outliers_test(X_test,clean)
        print("sets are cleaned")
        model=train_model(X_cleaned,y, n_estimators=350,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=21)
        print("model is trained")
        kaggle_submission(X_test_cleaned,model,"docs/Kaggle_submission.csv")

    if cleaning is True and scaling is True:
        scaler=set_scaling(X)
        X_scaled=data_scaling(scaler,X)
        print("trainingset is scaled")
        X_test_scaled=data_scaling(scaler,X_test)
        print("testset is scaled")
        X_cleaned_scaled,clean=clipping_outliers_train(X_scaled)
        X_test_cleaned_scaled=clipping_outliers_test(X_test_scaled,clean)
        model=train_model(X_cleaned_scaled,y, n_estimators=350,min_samples_split=2,min_samples_leaf=1,max_features='sqrt',max_depth=21)
        print("model is trained")
        kaggle_submission(X_test_cleaned_scaled,model,"docs/Kaggle_submission.csv")

    print("file is made with predictions")
    endtime=time.time()
    print("the model is trained en data is predicted")
    print("this took " + str(endtime-starttime) + " seconds")