#file for all code we aren't actively using
import numpy as np
import matplotlib.pyplot as plt


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

def clipping_graph():
    data={1:{'0': 2.587978249229756, '1': 2.5874567472918493, '2': 2.5966800476026988, '3': 2.596145257346921, '4': 2.6005530103281043, '5': 2.604303509834665}, 
            2:{'0': 2.589725623713276, '1': 2.5890236567773703, '2': 2.595744403109823, '3': 2.5970707612110693, '4': 2.598232377914856, '5': 2.6014873296917163}, 
            3:{'0': 2.600519924574632, '1': 2.5951610007545125,'2': 2.597199020716986, '3': 2.598025027289715, '4': 2.6037089507348914, '5': 2.6033617143851364}, 
            4:{'0': 2.607395235134827, '1': 2.60910703308158,'2': 2.5994255746311477, '3': 2.6022048958390926, '4': 2.6037478084112404, '5': 2.615830063456015}}
    markers = ['co-', 'ro-', 'go-', 'yo-']
    labels = ['1st ranked', '2nd ranked', '3rd ranked', '4th ranked']
    plt.xlabel('Clip size (percentile)')
    plt.ylabel('Mean absolute error')
    plt.title('Mean absolute error for different clip sizes for 4 highest-ranked options of encodings')
    for n in range(len(data)):
        x=(data[n+1].keys())
        y=list(data[n+1].values())
        plt.plot(x,y,markers[n],label=labels[n])
    plt.legend()
    plt.grid()
    plt.show()

def elbow_graph():
    x=[35,38,30,7,41,33,76,43,40,39,6,44,36,74,62,69,47,11,60,53,82,55,61,59,13,72,58,84,52,51,46,12,63,49,86,56,50,48,10,68,54,85,90,88,89,73,93,87,98,95,91,94,83,96,92,99,37,19,9,2,29,22,34,24,26,16,1,31,32,42,70,67,45,5,66,65,64,80,77,57,8,78,75,79,21,17,18,3,25,23,71,27,14,28,4,20,15,81,102,100,101,97,104,103,105]
    y=[2.60674322419264,2.60758200389518,2.60419929075543,2.55669616903978,2.61436525542661,2.60541009013401,2.6544432082437,2.61738728900623,2.61277406973663,2.60813216802147,2.55241334835406,2.61838046758399,2.6069395394369,2.65423124572268,2.64073825745369,2.64385895247479,2.62773571231567,2.58275606257297,2.63975177914009,2.63498391807791,2.66894094925405,2.63654908101676,2.6400838413151,2.6380420094577,2.59090259004142,2.65144144063985,2.63726815883421,2.67397976430801,2.634558153287,2.63181698950891,2.62555086250212,2.58359161910186,2.64109461332877,2.63100222976045,2.68274911444746,2.63661088663594,2.63161233814687,2.62993807169079,2.58216097384606,2.64256466880781,2.63501915747999,2.6813341064325,2.71159801988482,2.70442110101776,2.70443097451725,2.65297539698961,2.7156314484168,2.70389665929563,2.75469231791237,2.72696379587127,2.71412665438425,2.71870477339507,2.67374999300148,2.72904514989959,2.71444276273191,2.76744752719156,2.60744813893745,2.59344706338729,2.57989628972703,2.51403960445375,2.60135768078337,2.59403720761281,2.6066467336491,2.59587118742722,2.59834160953054,2.59152068867415,2.51094953316281,2.6044860653207,2.60470508908187,2.61448892145468,2.64687556835645,2.64163410305727,2.61858340731564,2.55097034658934,2.64150590621085,2.64118640516281,2.64114999731395,2.66179774760379,2.65602081721347,2.63683957100727,2.55978893643161,2.65741868443669,2.65427237393153,2.65858583406709,2.59356979122212,2.59232960155039,2.59259561398758,2.52945365221345,2.59589738659478,2.59415055206693,2.65117052371693,2.60012620860791,2.59140763951715,2.60047030639614,2.53266732744935,2.59354552753493,2.59144790656611,2.6675095349781,2.7931776193335,2.78972039581734,2.79046871318891,2.73907383210624,2.79534226539412,2.79374294109977,2.83244164742752]
    xticks=range(0,106,5)
    plt.plot(x,y,'bo')
    plt.title('MAE of different combinations of encodings')
    plt.xlabel('Ranking of model with this combination of encodings')
    plt.ylabel('Mean absolute error')
    plt.grid()
    plt.xticks(xticks)
    plt.xlim(0,107)
    plt.show()

def estimators_errors(encoding_bools, max_features=None, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    print('started calculating n_estimators')
    uniprot_dict=extract_sequence("data/protein_info.csv")
    smiles_train,uniprot_ids_train,y_train=data_to_SMILES_UNIProt_ID("data/train.csv")
    X = extract_true_features(encoding_bools, uniprot_dict, smiles_train, uniprot_ids_train)
    training_set, validation_set = train_validation_split(X,y_train,0.8)
    X_train, y_train = training_set
    X_validation, y_validation_true = validation_set
    print(f"data has been splitted into two sets")
    X_train_clipped, clip = clipping_outliers_train(X_train)
    X_validation_clipped = clipping_outliers_test(X_validation, clip)
    train_errors = []
    validation_errors = []
    n_estimators_range=range(150,551,50)
    for n_estimators in n_estimators_range:
        rf_model = train_model(X_train_clipped, y_train, criterion='squared_error',n_estimators=n_estimators, max_depth=max_depth, 
                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1)
        print(f"model has been trained with n_estimators {n_estimators}")
        train_error, validation_error = calculate_errors(X_train=X_train_clipped, y_train_true=y_train, X_validation=X_validation_clipped, y_validation_true=y_validation_true,
                     encoding_bools=encoding_bools, rf_model=rf_model, n_estimators=n_estimators,
                      max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, max_features=max_features)
        train_errors.append(train_error)
        validation_errors.append(validation_error)
    return train_errors, validation_errors, n_estimators_range

def estimators_graph():
    n_estimators=range(150,551,50)
    train_errors=[0.9364852503923449, 0.9387822832720553, 0.9368893137739037, 0.9338982757980732, 0.931374056099311, 0.9343564149344363, 0.9311238458740473, 0.9283287534314073, 0.9304546677103404]
    validation_errors=[2.4266153572854803, 2.403839310114793, 2.3976645726907426, 2.3916775985694687, 2.396581512252472, 2.3990959471799833, 2.387312014475291, 2.3891048845890626, 2.3988525188348513]
    # plt.plot(n_estimators, train_errors, 'bo-', label='Train MAE')
    plt.plot(n_estimators, validation_errors, 'ro-', label='Validation MAE')
    # plt.legend()
    plt.title('The validation error for different values of n_estimators')
    plt.xlabel('Value for n_estimators')
    plt.ylabel('MAE score on validation set')
    # plt.ylim(0.5,3)
    plt.grid()
    plt.show()
estimators_graph()