def extract_sequence(document):
    """input is a string that leads to the document, with there in
        first the uniprotid then the proteinacronym and last the one
        letter code sequence of the protein, split by comma's. the first
        line is to tell, which column is which. the output is a dictionary,
        with as keys the uniprotid and as value the one letter code 
        sequence of the protein."""
    file=open(document) #document needs to be in the right format
    lines=file.readlines()
    lines.pop(0)
    file.close()
    uniprot_dict={}
    for line in lines: 
        uniprot_id,protein_acronym,protein_sequence=line.split(",")
        protein_sequence=protein_sequence.strip().replace('"','')
        uniprot_id=uniprot_id.replace('"','')
        uniprot_dict[uniprot_id]=protein_sequence
    return uniprot_dict

data_dict=extract_sequence("data/protein_info.csv")
print(len(data_dict.keys()))
print(len(max(data_dict.values())))
print(len(min(data_dict.values())))