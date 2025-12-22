#file for all code we aren't actively using
import numpy as np


def combining_all_features(dictionary, affinity, ligandf=True, topological=True, morgan=True, 
                           macckeys=True, peptidef=True, windowbased=True, autocorrelation=True):
    """This functions makes an matrix with the descriptors from the ligands and proteins in the file
    
    Input: dictionary made in extract_all_features
    
    Output: matrix (n_samples*n_features) and affinity (np.array of length n_samples)
    """
    if ligandf==True:
        lf=dictionary['ligandf']
    if topological==True:
        tf=dictionary['topologicalf']
    if morgan==True:
        mo=dictionary['morganf']
    if macckeys==True:
        ma=dictionary['macckeysf']
    if peptidef==True:
        pf=dictionary['peptidef']
    if windowbased==True:
        wb=dictionary['windowbasedf']
    if autocorrelation==True:
        ac=dictionary['autocorrelationf']


    if ligandf==True and topological==True and morgan==True and macckeys==True:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,tf,mo,ma,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,tf,mo,ma,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,tf,mo,ma,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,tf,mo,ma,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([lf,tf,mo,ma,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,tf,mo,ma,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,tf,mo,ma,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==True and topological==True and morgan==True and macckeys==False:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,tf,mo,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,tf,mo,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,tf,mo,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,tf,mo,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([lf,tf,mo,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,tf,mo,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,tf,mo,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==True and topological==True and morgan==False and macckeys==True:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,tf,ma,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,tf,ma,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,tf,ma,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,tf,ma,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([lf,tf,ma,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,tf,ma,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,tf,ma,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==True and topological==False and morgan==True and macckeys==True:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,mo,ma,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,mo,ma,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,mo,ma,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,mo,ma,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([lf,mo,ma,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,mo,ma,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,mo,ma,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==False and topological==True and morgan==True and macckeys==True:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([tf,mo,ma,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([tf,mo,ma,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([tf,mo,ma,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([tf,mo,ma,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([tf,mo,ma,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([tf,mo,ma,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([tf,mo,ma,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==True and topological==True and morgan==False and macckeys==False:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,tf,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,tf,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,tf,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,tf,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([lf,tf,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,tf,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,tf,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==True and topological==False and morgan==True and macckeys==False:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,mo,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,mo,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,mo,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,mo,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([lf,mo,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,mo,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,mo,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==True and topological==False and morgan==False and macckeys==True:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,ma,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,ma,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,ma,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,ma,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([lf,ma,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,ma,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,ma,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==False and topological==True and morgan==True and macckeys==False:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([tf,mo,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([tf,mo,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([tf,mo,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([tf,mo,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([tf,mo,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([tf,mo,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([tf,mo,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==False and topological==True and morgan==False and macckeys==True:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([tf,ma,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([tf,ma,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([tf,ma,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([tf,ma,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([tf,ma,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([tf,ma,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([tf,ma,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==False and topological==False and morgan==True and macckeys==True:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([mo,ma,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([mo,ma,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([mo,ma,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([mo,ma,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([mo,ma,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([mo,ma,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([mo,ma,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==True and topological==False and morgan==False and macckeys==False:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([lf,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([lf,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([lf,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([lf,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==False and topological==True and morgan==False and macckeys==False:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([tf,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([tf,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([tf,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([tf,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([tf,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([tf,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([tf,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==False and topological==False and morgan==True and macckeys==False:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([mo,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([mo,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([mo,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([mo,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([mo,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([mo,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([mo,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==False and topological==False and morgan==False and macckeys==True:
            if peptidef==True and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([ma,pf,wb,ac],axis=1)
            elif peptidef==True and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([ma,pf,wb],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([ma,pf,ac],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==True:
                all_features=np.concatenate([ma,wb,ac],axis=1)
            elif peptidef==True and windowbased==False and autocorrelation==False:
                all_features=np.concatenate([ma,pf],axis=1)
            elif peptidef==False and windowbased==True and autocorrelation==False:
                all_features=np.concatenate([ma,wb],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==True:
                all_features=np.concatenate([ma,ac],axis=1)
            elif peptidef==False and windowbased==False and autocorrelation==False:
                raise RuntimeError("at least 1 peptide feature needs to be true")
    elif ligandf==False and topological==False and morgan==False and macckeys==False:
            raise RuntimeError("at least 1 ligand feature needs to be true")
                        

    matrix=all_features

    return matrix,affinity