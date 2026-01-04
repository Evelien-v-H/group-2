#Dit is het definitieve document, hier mag alleen code in die met pep 8 gestructureerd is en die in het definitieve document moet komen

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
        Returns the macc keys
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
        """This function returns an array with all sorts of descriptors gotten from rdkit
        
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
        """This function gets the topological fingerprints of an molecule
        
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
        """This function gets the MACCS-keys of an molecule
        
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
            An list with the autocorrelation encoding
        """
        n_residues, n_descr = np.shape(all_residue_descr)
        autocorrelation_features = []
        for descr in range(n_descr):
            current_descr_values = all_residue_descr[:, descr]
            descr_scaled = current_descr_values - np.mean(current_descr_values)
            correlation = np.correlate(descr_scaled, descr_scaled, mode='full')
            for lag in [1,4,10]:
                autocorrelation_features.append(correlation[lag])
        return autocorrelation_features  #a list of all autocorrelation_based features, length is 3*n_descr

def is_number(val):
    """Checks if the input a number is (used for datacleaning)"""
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False

def data_cleaning_train(data):
    """This function is used for datacleaning
    
    Cleans the data as it removes coloms without floats or integer. It replaces empty values or strings in features with the mean of 
    the feature, gives an error if een float or int is too big for np.float32 (and with this too big for the random forest) and it 
    returns the cleaned matrix and two lists for information about this process. Note that this process is meant for cleaning a 
    trainingset matrix.
    
    Parameters:
    --------------
    data: 2D (NumPy) array
    
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

        if len(values_colom)!=0:
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

