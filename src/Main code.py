import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

#relevant parts of rd_kit

#We need to implement this
# Aromaticity
# Different kinds of fingerprints
# Molecular Sanitzation --> berekend verschillende relevante waarden en kan kijken ofdat het molecuul uberhaupt wel klopt
#Stereo iets --> kan relevant zijn weet niet helemaal hoe dit werkt --> Je kan er ook mee checken ofdat iets erin zit
# Atropisomeric Bonds
# Molecular weight
    #--> rdkit.Chem.Descriptors.ExactMolWt(*x, **y)
    #--> The average molecular weight of the molecule ignoring hydrogens HeavyAtomMolWt(Chem.MolFromSmiles('CC'))
    #--> The average molecular weight of the molecule MolWt(Chem.MolFromSmiles('CC'))
#radial molecules Numradicalelectrons(Chem.MolFromSmiles('...'))
#valence elektrons NumValenceElectrons(Chem.MolFromSmiles('CC'))
#https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html
# Lipophilicity rdkit.Chem.Crippen.MolLogP(*x, **y) https://www.rdkit.org/docs/source/rdkit.Chem.Crippen.html
# Hydrogen rdkit.Chem.Lipinski.NHOHCount(x) https://www.rdkit.org/docs/source/rdkit.Chem.Lipinski.html
# weet niet ofdat dit werkt psa = rdMolDescriptors.CalcTPSA(mol) --> https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html
# Ik weet niet ofdat dit werkt rot_bonds_strict = rdMolDescriptors.CalcNumRotatableBonds(mol, strict=True) https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html


#Half relevant --> I don't know if we want to implement this
# There is information about how Smiles is used https://www.rdkit.org/docs/RDKit_Book.html#additional-information-about-the-fingerprints
# Atom-Atom Matching in Substructure Queries -->Je kan kijken of er structuren in de molecuul zitten
# Generic ("Markush") queries in substructure matching --> matcht groepen

#Snap ik niet helemaal/is denk niet helemaal relevant
# --> JSON support: I don't know what this is exactly, but I think it are two different ways for notating thinks and it can be
#usefull to know what what does, for some functions, but I don't think this is the most relevant information for now
# Self-Contained Structure Representations (SCSR) for Macromolecules --> geen idee hoe dit werkt, lijkt voor grote moleculen en is denk niet relevant voor kleine moleculen
# feature flags --> nuttig om te kijken wat het bij welke versies doet, maar 

# Meeste hiervan is relevant voor kleine moleculen, ik denk dat je voor grotere moleculen punt b ook prima met RDkit kan coderen. Heb hier nog geen research naar gedaan soms lastig zoek op die webstie

#Opmerkingen ik denk dat het handig is om te gaan kijken ofdat je een clas small molecule kan aanmaken en ene klas groot molecuul.
#Engels of Nederlandse peerreview
#advies voor hoe functies binnen de RDkit implementeren

#Openen data:
def open_data(datafile):
    """The files with data need to be imported.
    
    Input: a csv datafile
    
    Output: a dataFrame"""
    df=pd.read_csv(datafile)
    return df


def splitting_data_training(datafile):
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

    def rings_aromatic(self):
        """Calculates if and how many aromatic rings are in the structure. 
        
        Input: only self is needed (the self.molecule)
        
        Output: A number which is the number of aromatic rings"""
        aromatic_ring_count=0
        rings=self.molecule.GetRingInfo()
        atom_rings=rings.AtomRings()
        
        if not atom_rings:
            return aromatic_ring_count

        for ring in atom_rings:
            if all(self.molecule.GetAtomWithIdx(idx).GetIsAromatic()for idx in ring):
                aromatic_ring_count+=1
        return aromatic_ring_count
    
    def Topological_fingerprints(self):
        topfingergen = AllChem.GetRDKitFPGenerator()
        topfinger = topfingergen.GetFingerprint(self.molecule)
        array = np.zeros((topfinger.GetNumBits(),), dtype=int)
        DataStructs.ConvertToNumpyArray(topfinger, array)
        return array
    
    def morgan_fingerprint(self):
        morgan_fingergen = AllChem.GetMorganGenerator(radius=2)
        morganfingerprint = morgan_fingergen.GetSparseCountFingerprint(self.molecule)
        return morganfingerprint
    
    def macckeys(self):
        macckeys = MACCSkeys.GenMACCSKeys(self.molecule)
        array = np.zeros((macckeys.GetNumBits(),), dtype=int)
        DataStructs.ConvertToNumpyArray(macckeys, array)
        return array


    def get_all_descriptors(mol):
        """
        Returns a dictionary with all RDKit descriptors for a molecule.
    
        Input:
        mol : RDKit Mol object
        Output:
        dict : {descriptor_name: value}
        """
        descriptor_values = {}
    
        for name, func in Descriptors.descList:
            try:
                descriptor_values[name] = func(mol)
            except Exception as e:
                # Sommige descriptors kunnen fouten geven, we slaan deze over
                descriptor_values[name] = None
                print(f"Warning: {name} could not be calculated: {e}")
    
        return descriptor_values

    def rdkit_function(self):
        """Goal of this function is calculating all sort of information about the small molecule compound of the reaction.
        
        AANVULLEN!!!!:
        At the moment these calulated things are:
         
        
        Input: self, but with information from the SMILES string of the small molecule. Because this information is needed for coding with
        RDkit
         
        Output: Numpy array: 
        AANVULLEN!!

        """      
        #The number of aromatic rings
        number_of_aromatic_rings=self.rings_aromatic()

        #The formal charge of the molecule
        formal_charge=Chem.GetFormalCharge(self.molecule)

        #Molecular weight
        mol_weight = Descriptors.MolWt(self.molecule)
        
        #The lipophilicity
        logp = Crippen.MolLogP(self.molecule)

        #How many hydrogen donators there are in the molecule 
        hbonds_donors = Lipinski.NumHDonors(self.molecule)
        
        #How many hydrogen acceptors there are in the molecule
        hbonds_acceptors = Lipinski.NumHAcceptors(self.molecule)
        
        #Polar surface area
        polar_surface_area = rdMolDescriptors.CalcTPSA(self.molecule)
        
        #How many rotatable_bonds
        rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(self.molecule, strict=True)

        #Topological fingerprints
        topological_fingerprint= self.Topological_fingerprints()

        #Morgan fingerprints
        morganfingerprint=self.morgan_fingerprint()

        #macckeys
        macckey=self.macckeys()

        RDkit_array=np.array([number_of_aromatic_rings,formal_charge,mol_weight,logp,hbonds_donors,hbonds_acceptors,polar_surface_area,rotatable_bonds,topological_fingerprint,morganfingerprint,macckey])
        return RDkit_array


class proteins:
    def __init__(self, uniprot_id):
        self.uniprot_id = uniprot_id

    def extract_sequence(self):
        #Voor Iris: hier kan jouw extract_sequence ding komen

    def 

SMILES,UNIProt_ID,affinityscore=splitting_data_training('data/train.csv')
SMILE=SMILES[0]


for i in SMILES:
    Molecule=small_molecule(i)
    print(Molecule.rdkit_test())
   