import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import rdFingerprintGenerator

import peptidy as pep
from peptidy import *

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error



def data_to_SMILES_UNIProt_ID(datafile):
    """This function splits the dataset in a SMILES array and UNIProt_ID. If the datafile is in the format of the trainingset, it returns also the binding affinity score. 

        Parameters: 
        ----------------
        datafile: csv file with the format of either the given test or training data
            format of trainingset: first colom SMILES, second colom: UNIProt_ID, third colom: binding affinity score.
            format of testset: first colom numbers, second colom: SMILES, third colom: bindingaffinity score
    
        Returns: 
        ----------------
        SMILES: str

        UNIProt_ID: str

        affinity: str, float or int
            if the datafile is testset it is a string (with 'unknown affinity)
            if the datafile is trainingset it is a float or int
        """
    df=pd.read_csv(datafile)
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

def extract_sequence(document="data/protein_info.csv"):
    """Uses document (in this project "data/protein_info.csv) to extract an uniprotdictionary with information about the protein

    The dictionary that gets made is needed for the function extract_all_features and contains the key UNIPROT_ID and as value the one
    lettercode sequence of the protein.

    Parameters:
    ------------
    document: csv file
        format: first colom UNIProt_ID, second colom proteinacronym, third colom one letter code sequence of the protein
        
    
    Returns:
    -----------
    uniprot_dic: dict[str,str]
        key=UNIPROT_ID, value=one lettercode sequence of protein
    """
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
    """A class used to calculate the features for the ligand
    
    Attributes
    --------------
    SMILES: str
        A string which contains the SMILES string

    Methods
    -------------
    rdkit_descriptors()
        Returns the physiochemical descriptors of the molecule

    topological_fingerprints()
        Returns the topological fingerprint of the molecule

    morgan_fingerprint(radius=2,nBits=1024)
        Returns the morgan fingerprint of the molecule

    maccskeys()
        Returns the MACCS-keys
    """
    def __init__(self,SMILES):
        """
        Parameters:
        -------------
        SMILES: str
        """
        self.SMILES=str(SMILES)
        self.molecule=Chem.MolFromSmiles(str(SMILES))


    def rdkit_descriptors(self):
        """This function returns an array with all sorts of physiochemical descriptors gotten from rdkit
        
        Returns
        -----------
        array: (NumPy) array
        """
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
        """This function gets the topological fingerprints of a molecule
        
        Returns:
        ----------
        array: (NumPy) array
        """
        topfingergen = AllChem.GetRDKitFPGenerator(fpSize=2048)
        topfinger = topfingergen.GetFingerprint(self.molecule)
        array = np.zeros((topfinger.GetNumBits(),), dtype=int)
        DataStructs.ConvertToNumpyArray(topfinger, array)
        return array
    
    def morgan_fingerprint(self,radius=2,nBits=1024):
        """Returns a Morgan fingerprint as a NumPy array.

        Parameters:
        --------------
        radius: int
            The radius that defines the neighborhood size around each atom

        nBits: int
            The length of the fingerprint vector

        Returns:
        --------------
        array: (NumPy) array
        """
        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    
        # Generate the fingerprint (bit vector)
        fingerprint = morgan_generator.GetFingerprint(self.molecule)
    
        # Convert to NumPy array
        array = np.zeros((nBits,), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fingerprint, array)
    
        return array
    
    def maccskeys(self):
        """This function gets the MACCS-keys of a molecule
        
        Returns
        -----------
        array: (NumPy) array 
        """
        maccskeys = MACCSkeys.GenMACCSKeys(self.molecule)
        array = np.zeros((maccskeys.GetNumBits(),), dtype=int)
        DataStructs.ConvertToNumpyArray(maccskeys, array)
        return array

class protein:
    """A class used to calculate the features for the ligand
    
    Attributes
    --------------
    UNIProt_ID: str
        A string which contains the UNIProt_ID

    proteindict: dict
        A dictionary made with one lettercode sequence from proteins

    Methods
    -------------
    uniprot2sequence()
        Returns a string with the protein one letter code sequence

    sequence2onehot(sequence)
        Returns a list with the encoded peptide sequence represented as one-hot encoded vector

    extract_global_descriptors(sequence)
        Extracts the protein features from the amino acid sequence using peptidy

    extract_residue_descriptors(sequence)
        Estracts the residue descriptors using peptidy

    compute_window_based_features(sequence, all_residue_descr)
        Computes the window_based encoding, where the structure is more used then in global descriptors

    compute_autocorrelation_features(all_residue_descr)
        Computes the autocorrelation encoding, where the structure is more used then in global descriptors
    """
    def __init__(self, uniprot_id, proteindict):
        """
        Parameters:
        -------------
        uniprot_id: str
            The UNIProt_ID

        proteindict: dict
            The proteindict is the dictionary made in extract_sequence with the csv file 'protein_info.csv'
        """
        self.uniprot_id = uniprot_id
        self.dictionary=proteindict

    def uniprot2sequence(self):
        """This function returns a string with the protein one letter code sequence

        Returns:
        ---------------
        sequence: str
            The protein one letter code sequence
        """
        uniprot_dict=self.dictionary
        sequence=uniprot_dict[self.uniprot_id]
        return sequence #returns one letter code sequence of the protein
    
    def sequence2onehot(self,sequence):
        """Returns a list with the encoded peptide sequence represented as one-hot encoded vector

        Parameters:
        -------------
        sequence: str
            The sequence of a protein in one letter code. This is made in uniprot2sequence

        Returns:
        -------------
        onehot: list
            This is a list with the encoded peptide sequence represented as one-hot encoded
        """
        onehot=pep.one_hot_encoding(sequence,822) #longest protein is 822 amino acids
        return onehot
    
    def extract_global_descriptors(self, sequence):
        """Extracts the protein features from the amino acid sequence using peptidy. 
        
        Parameters:
        --------------
        sequence: str
            The sequence of a protein in one letter code. This is made in uniprot2sequence

        Returns:
        --------------
        peptidy_global_features_list: list
            List of all numerical features from peptidy of this protein
        """    
        peptidy_features_dict = pep.descriptors.compute_descriptors(sequence, descriptor_names=None, pH=7)
        peptidy_global_features_list = list(peptidy_features_dict.values())
        return peptidy_global_features_list
    
    def extract_residue_descriptors(self, sequence):
        """Extracts the residue descriptors using peptidy

        Parameters:
        --------------
        sequence: str
            The sequence of a protein in one letter code. This is made in uniprot2sequence
        
        Returns:
        ---------------
        array
            An array with residue descriptors
        """
        all_aa_descr = pep.encoding.aminoacid_descriptor_encoding(sequence, descriptor_names=None)
        return np.array(all_aa_descr)  #shape: (n_residues, n_descriptors)
    
    def compute_window_based_features(self, sequence, all_residue_descr):
        """Computes the window_based encoding, where the structure is more used then in global descriptors

        Parameters:
        ------------
        sequence: str
            The sequence of a protein in one letter code. This is made in uniprot2sequence

        all_residue_descr: array
            An array with residue descriptors made in the function extract_residue_descriptors

        Returns:
        --------------
        aggregated_window_descr: list
            An list with the window based encodings
        """
        n_residues, n_descr = np.shape(all_residue_descr)
        aggregated_window_descr = []
        for window_size in [4,8,15]:  #three different window sizes for short, medium, and long-range interactions
            for descriptor in range(n_descr):
                window_statistics = []
                for window_start in range(0, len(sequence), window_size):
                    window_stop = window_start + window_size
                    mean = np.mean(all_residue_descr[window_start:window_stop, descriptor])
                    variance = np.var(all_residue_descr[window_start:window_stop, descriptor])
                    window_statistics.append([mean, variance]) 
                window_statistics = np.array(window_statistics)  #shape: (n_residues/window_size , 2)

                for window_statistic in range(np.shape(window_statistics)[1]):
                    mean = np.mean(window_statistics[:, window_statistic])  #calculates the mean over all windows of each window statistic
                    sum = np.sum(window_statistics[:, window_statistic])
                    variance = np.var(window_statistics[:, window_statistic])
                    max_val = max(window_statistics[:, window_statistic])
                    aggregated_window_descr.extend([mean, sum, variance, max_val])
        
        return aggregated_window_descr  #a long list of all window-based protein descriptors, length = 24*n_descr

    def compute_autocorrelation_features(self, all_residue_descr):
        """Computes the autocorrelation encoding
        
        In this function the autocorrelation is computed for three different lags, all 
        having different biological relevance because they capture different ranges of interaction.
        
        Parameters:
        ------------
        all_residue_descr: array
            An array with residue descriptors made in the function extract_residue_descriptors
        
        Returns:
        -------------
        autocorrelation_features: list
            A list of length 3*n_descriptors with the autocorrelation encoding for this protein
        
        """
        n_residues, n_descr = np.shape(all_residue_descr)
        autocorrelation_features = []
        for descr in range(n_descr):
            current_descr_values = all_residue_descr[:, descr]
            descr_scaled = current_descr_values - np.mean(current_descr_values)                 #scale the descriptors to ensure correct correlation computation
            correlation = np.correlate(descr_scaled, descr_scaled, mode='full')
            zero_lag_ind = n_descr - 1                                                #np.correlate() returns an array of size 2*n_descr-1 with the zero-lag value in the middle
            for lag in [1,4,10]:
                autocorrelation_features.append(correlation[zero_lag_ind+lag])
        return autocorrelation_features                     #a list of all autocorrelation_based features, length is 3*n_descr

def create_tf_combinations(remaining, current):
    """Makes a list of lists with all possible combinations of True and False

    This function is used for testing all encodings

    Parameters:
    ---------------
    remaining : int
        remaining indicates the length of each list of booleans

    current : list of bool
        Current should always be [] for this algorithm to work.

    Returns:
    -----------------
    final_list: list
        list of lists of all possible combinations of True and False. 
    """
    if remaining == 0:
        return [current]
    with_true = create_tf_combinations(remaining - 1, current + [True])
    with_false = create_tf_combinations(remaining - 1, current + [False])
    final_list=with_true + with_false
    return final_list

def verify_tf_combinations(tf_combinations):
    """Checks if all true false combination list have at least one encoding from the ligand and one encoding from protein  set to true.
    
    Parameters:
    -------------
    tf_combinations: list
        list with all true false combinations
    
    Returns:
    -------------
    valid_tf_combinations: list
        list with all valid true and false combinations
    """
    valid_tf_combinations = []
    for list_of_bools in tf_combinations:
        if (list_of_bools[0] or list_of_bools[1] or list_of_bools[2] or list_of_bools[3]) and (list_of_bools[4] or list_of_bools[5] or list_of_bools[6]):
            valid_tf_combinations.append(list_of_bools)
    return valid_tf_combinations

def extract_all_features(datafile, encoding_names):
    """This function makes a dictionary with all the different possible datasets. This function should only be used in combination with slicing_features().

    Parameters:
    ----------------
    Datafile: csv-file 
        A csv-file with a format of the trainingset or testset.
    Encoding_names: list 
        list of all possible encodings

    Returns:
    -----------------
    Dictionary: dict
        a dictionary with as key the name of the encoding and as value an array shape n_samples, n_features.

    Affinity: (Numpy) array
        1D array of shape (n_samples,) that contains the affinity per sample, as read from the datafile. If datafile does not 
        contain affinity (in the case of unlabeled data), affinity contain 'unknown affinity' for each sample.
    """
    SMILES,UNIProt_ID,affinity=data_to_SMILES_UNIProt_ID(datafile)
    uniprot_dict=extract_sequence("data/protein_info.csv")
    dictionary={}
    ligand_list=[]
    topological_list=[]
    morgan_list=[]
    maccskeys_list=[]
    peptide_list=[]
    windowbased_list=[]
    autocorrelation_list=[]

    for i in range(len(SMILES)):
        ligand=small_molecule(SMILES[i])
        peptide=protein(UNIProt_ID[i], uniprot_dict)
        sequence=peptide.uniprot2sequence()
        all_residue_descr=peptide.extract_residue_descriptors(sequence)

        ligand_features=ligand.rdkit_descriptors()
        ligand_topological=ligand.topological_fingerprints()
        ligand_morgan=ligand.morgan_fingerprint()
        ligand_maccskeys=ligand.maccskeys()
        peptide_features_list=peptide.extract_global_descriptors(sequence)
        peptide_windowbased_list=peptide.compute_window_based_features(sequence,all_residue_descr)
        peptide_autocorrelation_list=peptide.compute_autocorrelation_features(all_residue_descr)
        
        ligand_list.append(ligand_features)
        topological_list.append(ligand_topological)
        morgan_list.append(ligand_morgan)
        maccskeys_list.append(ligand_maccskeys)
        peptide_list.append(np.array(peptide_features_list))
        windowbased_list.append(np.array(peptide_windowbased_list))
        autocorrelation_list.append(np.array(peptide_autocorrelation_list))
    
    for encoding_name in encoding_names:
        if encoding_name=='ligandf':
            array=np.array(ligand_list)
        if encoding_name=='topologicalf':
            array=np.array(topological_list)
        if encoding_name=='morganf':
            array=np.array(morgan_list)
        if encoding_name=='maccskeysf':
            array=np.array(maccskeys_list)
        if encoding_name=='peptidef':
            array=np.array(peptide_list)
        if encoding_name=='windowbasedf':
            array=np.array(windowbased_list)
        if encoding_name=='autocorrelationf':
            array=np.array(autocorrelation_list)

        n_features=array.shape[1]
        dictionary[encoding_name]=array,n_features
    
    return dictionary,affinity



def extract_true_features(encoding_bools_dict, uniprot_dict, SMILES, UNIProt_ID):
    """Creates an 2D array that can be used for training and predictions of the model. 

    Parameters:
    -------------
    encoding_bools_dict: dict 
        contains for each possible encoding a boolean that indicates whether this encoding will be included in the array.
    uniprot_dict: dict 
        contains the sequence for each uniprot_id, which is needed for the initialisation of objects of class protein
    SMILES: (NumPy) array
        contains the smiles-strings for all samples, shape (n_samples,)
    UNIProt_ID: (NumPy) array
        contains the uniprot_ids for all samples, shape (n_samples,)
    
    Returns: 
    --------------
        X: 2D (NumPy) array 
            (n_samples, n_features) that contains all features from the encodings indicated by encoding_bools_dict
        """
    ligand_list=[]
    topological_list=[]
    morgan_list=[]
    maccskeys_list=[]
    peptide_list=[]
    windowbased_list=[]
    autocorrelation_list=[]

    for i in range(len(SMILES)):
        ligand=small_molecule(SMILES[i])
        peptide=protein(UNIProt_ID[i], uniprot_dict)
        sequence=peptide.uniprot2sequence()
        if encoding_bools_dict['windowbasedf'] or encoding_bools_dict['autocorrelationf']:
            all_residue_descr=peptide.extract_residue_descriptors(sequence)
        if encoding_bools_dict['ligandf']:
            ligand_features=ligand.rdkit_descriptors()
            ligand_list.append(ligand_features)
        if encoding_bools_dict['topologicalf']:
            ligand_topological=ligand.topological_fingerprints()
            topological_list.append(ligand_topological)
        if encoding_bools_dict['morganf']:
            ligand_morgan=ligand.morgan_fingerprint()
            morgan_list.append(ligand_morgan)
        if encoding_bools_dict['maccskeysf']:
            ligand_maccskeys=ligand.maccskeys()
            maccskeys_list.append(ligand_maccskeys)
        if encoding_bools_dict['peptidef']:
            peptide_features_list=peptide.extract_global_descriptors(sequence)
            peptide_features=np.array(peptide_features_list)
            peptide_list.append(peptide_features)
        if encoding_bools_dict['windowbasedf']:
            peptide_windowbased_list=peptide.compute_window_based_features(sequence,all_residue_descr)
            peptide_windowbased=np.array(peptide_windowbased_list)
            windowbased_list.append(peptide_windowbased)
        if encoding_bools_dict['autocorrelationf']:
            peptide_autocorrelation_list=peptide.compute_autocorrelation_features(all_residue_descr)
            peptide_autocorrelation=np.array(peptide_autocorrelation_list)
            autocorrelation_list.append(peptide_autocorrelation)
    
    true_list = []          #contains as items all encodings that have been set to True in encoding_bools_dict

    for encoding_name, encoding_bool in encoding_bools_dict.items():
        if encoding_bool:
            if encoding_name=='ligandf':
                array=np.array(ligand_list)
            if encoding_name=='topologicalf':
                array=np.array(topological_list)
            if encoding_name=='morganf':
                array=np.array(morgan_list)
            if encoding_name=='maccskeysf':
                array=np.array(maccskeys_list)
            if encoding_name=='peptidef':
                array=np.array(peptide_list)
            if encoding_name=='windowbasedf':
                array=np.array(windowbased_list)
            if encoding_name=='autocorrelationf':
                array=np.array(autocorrelation_list)
            true_list.append(array)
    
    X = np.concatenate(true_list, axis=1)

    return X

def slicing_features(large_feature_array, n_features_list, bool_list):
    """Slices the matrix with all possible encodings into the right combination of encodings.
    Parameters:
    ------------
    large_feature_array: 2D (NumPy) array
        array of shape (n_samples, total_n_features) with all possible small molecule encodings and protein encodings, in the following order: 
        ligand features, topological fingerprints, morgan fingerprints, maccskeys, peptide features (for the protein as a whole), 
        window-based features, and autocorrelation features.
    
    n_features_list: list 
        list of the number of features corresponding to each of the feature types described above, in the same order.
    
    bool_list: list 
        contains the booleans corresponding to each feature, in the same order.
    
    Returns:
    -------------
    sliced_features_array: 2D (NumPy) array 
        np.array of shape (n_samples, n_features) that consists of all features of which the boolean input parameter was set to True.
    """                               
    #keeps track of the encodings included in the output
    cumulative_n_features = np.cumsum(n_features_list)
    sliced_features=[]

    for i in range(len(bool_list)):
        if bool_list[i]:
            stop_index = cumulative_n_features[i]
            start_index = cumulative_n_features[i] - n_features_list[i]
            array_to_be_added = large_feature_array[:,start_index:stop_index]
            sliced_features.append(array_to_be_added)
    sliced_features_array = np.concatenate(sliced_features, axis=1)
    return sliced_features_array

def is_number(val):
    """Checks if the input a number is (used for datacleaning)"""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def data_cleaning_train(data, usefull_colom_percentage=20):
    """This function is used for datacleaning
    
    Cleans the data as it removes coloms without floats or integer. It replaces empty values or strings in features with the mean of 
    the feature, gives an error if een float or int is too big for np.float32 (and with this too big for the random forest) and it 
    returns the cleaned matrix and two lists for information about this process. Note that this process is meant for cleaning a 
    trainingset matrix.
    
    Parameters:
    --------------
    data: 2D (NumPy) array

    usefull_colom_percentage: int
        This is the percentage how much of an feature needs to be integers or floats. If an feature has less values, it deletes the feature.
        This percentage needs to be between 1-100
    
    Returns:
    --------------
    data: 2D (NumPy) array
        The cleaned dataset

    mean_value_coloms: list
        An list with information about the mean_values of all existing (cleaned) features
    
    irrelevant_colums: list
        An list with information about which coloms are deleted during the cleaning
    
    Raises:
    -------------
    ValueError
        If the value of the datapoint is to big to use in an random forest (it is bigger than np.float32). These problems need to
        be solved manually.
    """
    n_samples,n_features=np.shape(data)
    minimal_length_colom=(usefull_colom_percentage*n_samples)/100  
    irrelevant_colums=[]
    mean_value_coloms=[]
    #This functions cleans colom after colom, this results in a quite complex for-loop. The first for-loop selects the colom, 
    #the second for-loop selects the row and makes the mean value list and irrelevant colom list. The third for-loop selects the row
    #and checks if the datapoint needs to be cleaned
    for i in range(data.shape[1]):
        values_colom=[]
        for j in range(data.shape[0]):
            if is_number(data[j,i]) is True:
                values_colom.append(float(data[j,i]))

        if len(values_colom)>minimal_length_colom:
            mean_value=np.mean(values_colom)
            mean_value_coloms.append(mean_value)
                    
        else:
            irrelevant_colums.append(i)

        for k in range(data.shape[0]):
            if i not in irrelevant_colums:
                if is_number(data[k,i]) is True:
                    if float(data[k,i])>=np.finfo(np.float32).max:
                        raise ValueError("Youre value is to big for the float32 of the random forest. Solve this manually")
                    else:
                        data[k,i]=float(data[k,i])
            
                else:
                    data[k,i]=mean_value

    #In this part the irrelevant coloms get deleted. This is done after the colom cleaning because otherwise the indexes are messed up
    if len(irrelevant_colums)>0:
        irrelevant_colums.reverse()
        for n in irrelevant_colums:
            print('a feature is deleted colom:',n)
            data = np.delete(data, n, axis=1)
    return data, mean_value_coloms, irrelevant_colums

def data_cleaning_test(data,mean_value_coloms, irrelevant_colums):
    """This code cleans the test data with information from the cleaning of the trainingset
    
    Parameters:
    --------------
    data: 2D (NumPy) array

    mean_value_coloms: list
        An list with information about the mean_values of all existing (cleaned) features
    
    irrelevant_colums: list
        An list with information about which coloms are deleted during the cleaning
    
    Returns:
    --------------
    data:2D (NumPy) array

    Raises:
    -------------
    ValueError
        If the value of the datapoint is to big to use in an random forest (it is bigger than np.float32). These problems need to
        be solved manually.
    """
    #The irrelevant coloms are deleted. This is done before the colom cleaning because otherwise the indexes are messed up
    if len(irrelevant_colums)>0:
         for n in irrelevant_colums:
            print('a feature is deleted colom:',n)
            data = np.delete(data, n, axis=1)
    #The rest of the data is checked. First for-loop selects the colom, the second the row
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if is_number(data[j,i]) is True:
                if float(data[j,i])>=np.finfo(np.float32).max:
                    raise ValueError("Youre value is to big for the float32 of the random forest. Solve this manually")
                else:
                    data[j,i]=float(data[j,i])
            else:
                data[j,i]=mean_value_coloms[i]
    return data

def clipping_outliers_train(matrix,percentile_low=1,percentile_high=99):
    """This function clips outliers. The outliers get respectively the lowest and highest not outlier values. 

    This function is meant for the trainingset

    Parameters:
    ------------
    matrix:2D (NumPy) array

    percentile_low: int
        Every value under this percentile is considered an outlier and is turned into the lowest possible not outlier value

    percentile_high: int
        Every value above this percentile is considered an outlier and is turned into the highest possible not outlier value
    
    Returns:
    ------------
    matrix_output: 2D (NumPy) array
        This array has the same format as the input matrix
    
    percentile_list: list
        This list is needed for the test clipping and contains the values corresponding to the lowest percentile and high percentile in a tuple for each colom.
    """
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
    """This function clips outliers. The outliers get respectively the lowest and highest not outlier values. 

    This function is meant for the testset
    
        Parameters:
    ------------
    matrix:2D (NumPy) array

    percentile_list: list
        This list is needed for the test clipping and contains the values corresponding to the lowest percentile and high percentile in a tuple for each colom.
    
    Returns:
    -----------
    matrix_output: 2D (NumPy) array
        This array has the same format as the input matrix
    """
    new_array_list=[]
    for i in range (matrix.shape[1]):
        array=matrix[:,i]
        lowest_percentile,highest_percentile=percentile_list[i]
        output_array=np.clip(array,a_min=lowest_percentile,a_max=highest_percentile)
        new_array_list.append(output_array)
    
    matrix_output=np.column_stack(new_array_list)
    return matrix_output

def set_scaling(matrix):
    """Creates and fits a MinMaxScaler on the given datase. 
    
    Parameters:
    -------------
    matrix:2D (NumPy) array

    Returns:
    ------------
    scaler : sklearn.preprocessing.MinMaxScaler
        Fitted MinMaxScaler object
    """
    scaler=sklearn.preprocessing.MinMaxScaler()
    scaler.fit(matrix)
    return scaler

def data_scaling(scaler, data):
    """Transforms data from fixed scalar.

    Parameters:
    -------------
    scaler:sklearn.preprocessing.MinMaxScaler
        Fitted MinMaxScaler object (made in set_scaling)

    data: 2D (NumPy) array
        This data array needs to be an array with the same format as the matrix in set_scaling

    Returns:
    -----------
    data_scaled: 2D (NumPy) array
        This array has the same format as the input data
    """
    data_scaled=scaler.transform(data)
    return data_scaled

def fit_PCA(X, n_components=None):
    """Performs a PCA on the data in X 
    
    Parameters:
    -----------------
    X: 2D (NumPy) array
    
    n_components: int or none
        Decides how many n_components it keeps, if None it keeps all components

    Returns:
    ---------------
    X_scores: 2D (NumPy) array
        array of shape (n_samples, n_components)

    variance_per_pc: list
        list of ratio of variance explained per principal component
    """
    pca = sklearn.decomposition.PCA(n_components=n_components)
    X_scores = pca.fit_transform(X)
    variance_per_pc = pca.explained_variance_ratio_
    return X_scores, variance_per_pc


def select_principal_components(X_pca_scores, variance_explained, goal_cumulative_variance):
    """Selects principal components based on how many variance it explains

    From the input array X_pca_scores, creates a new array relevant_principle_components, which is a subset
    of the input array that includes only the relevant PCs to reach the goal_cumulative_variance. 

    Parameters:
    ----------------
    X_pca_scores: 2D (NumPy) array

    variance_explained: 2D (NumPy) array 
        variance is a np.array of shape (n_principal_components)

    goal_cumulative_variance: int or float
        The variance the PCA score needs to explain
    
    Returns:
    -------------
    relevant_principal_components:2D (NumPy) array
        All principal components that contains the portion of variance explained by each principal component."""
    cumulative_variance = 0
    pc = 0
    while cumulative_variance < goal_cumulative_variance:
        cumulative_variance += variance_explained[pc]
        pc += 1
    relevant_principal_components = X_pca_scores[:, :pc]
    return relevant_principal_components


def train_model(X,y,n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=1.0):
    """Trains the random forest model

    Parameters:
    ---------------
    X: 2D (NumPy) array
        This array needs to be the trainingdata

    y: (NumPy) array
        This array needs to be the corresponding affinity score.

    n_estimators: int
        n_estimators are the number of trees used

    Max_depth: int or None
        Max_depth is the maximium depth of the tree

    min_samples_split: int or float
        min_samples_split are how many samples are needed per split

    min_samples_leaf: int or float
        min_samples_leaf are the number of samples that are needed for a final leaf

    max_features: str
        max_features are how many feature are used to make a tree. The possible strings are sqrt, log2 and None

    Returns:
    ------------
    random_forest: a random forest model  
    """
    
    random_forest=RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                                        min_samples_leaf=min_samples_leaf, max_features=max_features)
    random_forest.fit(X,y)
    return random_forest 


def RF_predict(model, X_test):
    """This function uses a defined model to predict the y values of X_test. 
    
    Parameters:
    ------------
    model: a random forest model
        (made in train_model)

    X_test: 2D (NumPy) array
        X_test needs to have the same format as the training data used
    
    Returns:
    ------------
    y_values: (NumPy) array
    """
    y_values=model.predict(X_test)
    return y_values

def RF_error(model, X_test, y_test):
    """Uses R2 to calculate the error of the random forest model. 
    
    Parameters:
    ------------
    model: a random forest model 
        (made in train_model)

    X_test:2D (NumPy) array
        X_test needs to have the same format as the training data used

    y_test:(Numpy) array
        Needs to be the length of the number of samples in X_test
    
    Returns:
    ---------
    score: float
        This is the score of the model. The higher the score is, the better the model works.
    """
    score=model.score(X_test,y_test)
    return score

def data_prep_cv(data_prep_dict,affinity, data_prep_scores, n_estimators=100, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, max_features=None):
    """Performs cross-validation on the different data sources in data_prep_dict. 
    
    Parameters:
    --------------
    data_prep_dict: dict
        The data sources it does the crossvalidation on. These can for example be functionalities like scaling and clipping 
        outliers included or excluded.(made in make_data_prep_dict)

    affinity: (NumPy) array
        binding affinity per sample

    data_prep_scores: dict
        dictionary with as keys the same strings as data_prep_dict and as value a list of the cv scores achieved for that
        data prepping method. For each call of data_prep_cv, one value is appended to the end of each of the lists. Is used 
        to keep track of the overall score of each prepping method.

    n_estimators: int
        n_estimators are the number of trees used

    Max_depth: int or None
        Max_depth is the maximium depth of the tree

    min_samples_split: int or float
        min_samples_split are how many samples are needed per split

    min_samples_leaf: int or float
        min_samples_leaf are the number of samples that are needed for a final leaf

    max_features: str
        max_features are how many feature are used to make a tree. The possible strings are sqrt, log2 and None
    
    Returns:
    ------------
    best_dataprep: str
        key from data_prep_dict whose X_train array resulted in the lowest MAE during cross validation.

    best_cv_score: float
        mean squared error of the best_dataprep

    data_prep_scores: dict
        updated data_prep_scores, now having appended cv scores from the most recent call of this function to the
        lists in the values.

    """
    best_cv_score = 100
    for current_prep_name, current_X_train in data_prep_dict.items():                            #loops over the different data sources in the dictionary, data_source is the index of the current iteration
        score_list_current_prep = data_prep_scores[current_prep_name]           #a list of the scores of this data prepping
        estimator = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, 
                                          min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1, verbose=0)
        neg_mean_cv_score = cross_val_score(estimator, current_X_train, affinity, n_jobs=-1, cv=3, scoring='neg_mean_absolute_error', verbose=3).mean()
        mean_cv_score = -neg_mean_cv_score
        score_list_current_prep.append(mean_cv_score)
        if mean_cv_score < best_cv_score:        
            best_cv_score = mean_cv_score
            best_dataprep = current_prep_name          #keeps track of the best data prep thus far
        data_prep_scores[current_prep_name] = score_list_current_prep
    return best_dataprep, best_cv_score, data_prep_scores

def make_data_prep_dict(X_train_raw,include_only_cleaning,include_scaling=True,include_clipping=True,include_PCA='clipping'):
    """applies different data preppings to X_train_raw, depending on what boolean parameters have been set to true.
    
    Parameters:
    --------------------------
    X_train_raw 2D (NumPy) array 
        array of shape (n_samples, n_features).
    
    include_only_cleaning: boolean
        Determines whether only cleaned X_train_raw will be included
    
    include_scaling: boolean
        Determines whether scaling (min-max) will be applied.
    
    include_clipping: boolean 
        Determines whether the outliers will be clipped (values beyond 1st and 99th percentile replaced with values at 1st and 99th percentile). 
        If both scaling and clipping are True, clipping will be applied before scaling.
    
    include_PCA: False, 'no_clipping', 'clipping'
        Determines whether PCA will be applied. If False, will be excluded. If 'no_clipping', will be included without clipping. If 'clipping', will be included with clipping. If included, 
            three additional data prep options will be included in the data_prep_dict, that respectively explain 66%, 80%, and 95% of variance. When PCA is True, scaling will always be applied.
    
    Returns:
    --------------------
    data_prep_dict: dict 
        Dictionary with as keys the type of data prepping and as values the respective X_train array.
    """
    data_prep_dict={}
    X_train, mean_value_coloms, irrelevant_colums=data_cleaning_train(X_train_raw)
    if include_only_cleaning: data_prep_dict['X_only_cleaned'] = X_train
    if include_clipping:
        X_clipped, percentile_list = clipping_outliers_train(X_train)
        if not include_scaling: data_prep_dict['X_clipped'] = X_clipped
    elif include_scaling:
        scaler = set_scaling(X_train)
        X_scaled = data_scaling(scaler, X_train)
        data_prep_dict['X_scaled'] = X_scaled
    if include_clipping and include_scaling:
        X_clipped
        scaler = set_scaling(X_clipped)
        X_clipped_scaled = data_scaling(scaler, X_clipped)
        data_prep_dict['X_clipped_scaled'] = X_clipped_scaled
    if include_PCA=='no_clipping':
        X_train_pc_scores, variance_per_pc = fit_PCA(X_scaled)
        X_train_pca66 = select_principal_components(X_train_pc_scores, variance_per_pc, 0.66)
        X_train_pca80 = select_principal_components(X_train_pc_scores, variance_per_pc, 0.80)
        X_train_pca95 = select_principal_components(X_train_pc_scores, variance_per_pc, 0.95)
        data_prep_dict['Scaled+pca66']=X_train_pca66
        data_prep_dict['Scaled+pca80']=X_train_pca80
        data_prep_dict['Scaled+pca95']=X_train_pca95
    elif include_PCA=='clipping':
        X_train_clipped_pc_scores, variance_per_pc_cleaned = fit_PCA(X_clipped_scaled)
        X_train_clipped_pca66 = select_principal_components(X_train_clipped_pc_scores, variance_per_pc_cleaned, 0.66)
        X_train_clipped_pca80 = select_principal_components(X_train_clipped_pc_scores, variance_per_pc_cleaned, 0.80)
        X_train_clipped_pca95 = select_principal_components(X_train_clipped_pc_scores, variance_per_pc_cleaned, 0.95)
        data_prep_dict['Clipped+scaled+pca66']=X_train_clipped_pca66
        data_prep_dict['Clipped+scaled+pca80']=X_train_clipped_pca80
        data_prep_dict['Clipped+scaled+pca95']=X_train_clipped_pca95

    return data_prep_dict

def hyperparams_cv(X,y,param_grids, n_iter=100, cv_fold=5, search_type='randomized', scoring='neg_mean_absolute_error'):
    """Tunes the hyperparameters for the RF model using randomised search.
    
    Parameters:
    ---------------
    X: 2D (NumPy) array:
        array of size (n_samples * n_features)

    y: (NumPy) array
        array of size (n_samples)

    param_grids: dict 
        Contains the parameters that will be tuned and their grid of values that will be tried
    
    n_iter: int 
        Number of iterations the model will take, only relevant if search_type='randomized'
    
    cv_fold: int 
        Determines the fold of the cross validation, i.e. how many different predictions will be made per parameter combination
    
    Search_type: str ('randomized' or 'grid') 
        determines whether randomized or grid search will be performed.
    
    Returns:
    ---------------
    best_params:dict
        Dictionary of the most optimal parameters

    best_score:dict
        Dictionary of the most optimal score

    best_estimator:dict  
        Dictionary of the most optimal dictionary
    """
    model = RandomForestRegressor()
    if search_type=='grid':
        estimator = GridSearchCV(model, param_grids, n_jobs=-1, refit=True, cv=cv_fold, verbose=3, scoring=scoring)
    elif search_type=='randomized':
        estimator = RandomizedSearchCV(model, param_grids, n_jobs=-1, refit=True, cv=cv_fold, n_iter=n_iter, verbose=2, scoring=scoring)
    estimator.fit(X,y)
    best_estimator = estimator.best_estimator_
    best_params = estimator.best_params_
    best_score = estimator.best_score_
    return best_params, best_score, best_estimator


def calculate_errors(X_train, y_train_true, X_validation, y_validation_true,
                     encoding_bools, rf_model, n_estimators=None, max_depth=None, 
                     min_samples_split=None, min_samples_leaf=None, max_features=None, params=None):
    y_train_pred = RF_predict(rf_model, X_train)        #the predicted values of y_train
    y_validation_pred = RF_predict(rf_model, X_validation)    #the predicted values of y_validation
    mae_train = mean_absolute_error(y_train_true, y_train_pred)
    mae_validation = mean_absolute_error(y_validation_true, y_validation_pred)
    print(f'For encoding_bools: {encoding_bools}')
    if params is None:
        print(f"""and parameters: n_estimators: {n_estimators}, max_depth: {max_depth}, 
          min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, max_features: {max_features}""")
    if params is not None:
        print(f"and parameters: {params}")
    print(f'the MAE on the train set is: {mae_train} and on the validation set: {mae_validation}')
    return mae_train, mae_validation

