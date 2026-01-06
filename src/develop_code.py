import matplotlib as plt
import numpy
import time
from main_code import *


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

def kaggle_submission(X_test,model,filename):
    """This function makes the document needed for the kaggle submissions"""
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
    Parameter: pca_scores (np.array): the data transformed onto the new PCA feature space.
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

def make_pca_plots(pca_scores, y_train,cmap):
    """makes three PCA-plots: first vs second PC, first vs third PC, and second vs third PC. 
    Parameter: pca_scores (np.array): the data transformed onto the new PCA feature space."""
    fig, (ax1,ax2,ax3) = plt.subplots(3)
    fig.tight_layout()
    print(f'The shape of pca_scores[:,0] is {np.shape(pca_scores[:,0])}')
    fig.suptitle('Principal component plots on cleaned and scaled training data')
    vmin = np.percentile(y_train,1)
    vmax = np.percentile(y_train,99)
    ax1.scatter(pca_scores[:,0],pca_scores[:,1],s=0.5, c=y_train, cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set(xlabel='PC1',ylabel='PC2')
    ax2.scatter(pca_scores[:,0],pca_scores[:,2],s=0.5, c=y_train, cmap=cmap,vmin=vmin,vmax=vmax)
    ax2.set(xlabel='PC1',ylabel='PC3')
    ax3.scatter(x=pca_scores[:,1],y=pca_scores[:,2], s=0.5, c=y_train, cmap=cmap,vmin=vmin, vmax=vmax)
    ax3.set(xlabel='PC2',ylabel='PC3')
    plt.show()      


def clipping_graph():
    """
    Plots the data from the cross validation of the four best combinations of encodings with clipping sizes 0-5.
    """
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
    """
    Plots the graph of the ranked combinations of encodings vs their MAE. x and y hard-coded at the beginning of this function
    was output from the cross-validation experiment where all 105 possible combinations of encodings were compared.
    """
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
    """Computes the error on train and validation set for different values of n_estimators.
    """
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
    """Plots the graph of the train and validation error for different values of n_estimators. The data specified at the beginning
    of this function comes from running the estimators_errors function.
    """
    n_estimators=range(150,551,50)
    train_errors=[0.9364852503923449, 0.9387822832720553, 0.9368893137739037, 0.9338982757980732, 0.931374056099311, 0.9343564149344363, 0.9311238458740473, 0.9283287534314073, 0.9304546677103404]
    validation_errors=[2.4266153572854803, 2.403839310114793, 2.3976645726907426, 2.3916775985694687, 2.396581512252472, 2.3990959471799833, 2.387312014475291, 2.3891048845890626, 2.3988525188348513]
    f, (ax1, ax2) = plt.subplots(2,1,sharex=True,height_ratios=[15,1])
    # plt.plot(n_estimators, train_errors, 'bo-', label='Train MAE')
    ax1.plot(n_estimators, validation_errors, 'ro-', label='Validation MAE')
    # plt.legend()
    f.suptitle('The validation error for different values of n_estimators')
    f.subplots_adjust(hspace=0.1)
    f.supxlabel('Value for n_estimators')
    f.supylabel('MAE score on validation set')
    ax1.set_ylim(2.385,2.430)
    ax2.set_ylim(0,0.01)
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()
    ax1.grid()
    ax2.grid()
    ax2ygridlines = ax2.get_ygridlines()
    ax2ygridlines[-1].set_visible(False)
    plt.show()

def prepping_graph():
    data={'Scaled':0.4785018435920147,'Cleaned':0.47583234137174646,'Cleaned+\nscaled': 0.4784814897780814, 
          'Scaled+\npca66':0.3201183220520327, 'Scaled+\npca80':0.34864618071310116, 'Scaled+\npca95': 0.365840455772483,
          'Cleaned+\nscaled+\npca66':0.320365179686796, 'Cleaned+\nscaled+\npca80':0.3511814161572563, 'Cleaned+\nscaled+\npca95':0.3511814161572563}
    plt.xlabel('Preproccessing steps')
    plt.ylabel('R^2')
    plt.title('Error for different preprocessing steps',fontsize=18)
    plt.bar(list(data.keys()), list(data.values()))
    plt.show()
prepping_graph()

def tuning_clipping():
    """code used to tune the clip size"""
    starttime=time.time()
    bestscore=100
    print("started tuning clippings")
    uniprot_dict=extract_sequence("data/protein_info.csv")  
    smiles_train,uniprot_ids_train,y_train=data_to_SMILES_UNIProt_ID("data/train.csv")
    encoding_bools_list = [{'ligandf':False, 'topologicalf':True, 'morganf': True, 'maccskeysf': False, 
                    'peptidef': True, 'windowbasedf': False, 'autocorrelationf': False}, 
                    {'ligandf':False, 'topologicalf':True, 'morganf': True, 'maccskeysf': True, 
                    'peptidef': True, 'windowbasedf': False, 'autocorrelationf': False},
                    {'ligandf':False, 'topologicalf':False, 'morganf': True, 'maccskeysf': True, 
                    'peptidef': True, 'windowbasedf': False, 'autocorrelationf': False},
                    {'ligandf':False, 'topologicalf':False, 'morganf': True, 'maccskeysf': False, 
                    'peptidef': True, 'windowbasedf': False, 'autocorrelationf': False}]
    n_estimators=300
    max_depth=None
    min_samples_split=2
    min_samples_leaf=1
    max_features=None
    for encoding_bools in encoding_bools_list:
        X_train = extract_true_features(encoding_bools, uniprot_dict, smiles_train, uniprot_ids_train)
        print(f'data array has been made, this took {(time.time()-starttime)/60} minutes')
        X_train, mvl, irl = data_cleaning_train(X_train)
        print(f'data has been cleaned')
        clipped_dict={}
        clipped_dict['no clipping']=X_train
        for clip_size in range(1,6):
            X_train_clipped,clean=clipping_outliers_train(X_train, clip_size, 100-clip_size)
            clipped_dict[clip_size]=X_train_clipped
        clip_scores = {}
        for clip in list(clipped_dict.keys()):
            clip_scores[clip]=[]
        score, best_clip, clip_scores = data_prep_cv(clipped_dict, y_train, clip_scores, n_estimators, max_depth, 
                                                        min_samples_split, min_samples_leaf, max_features)  
        print(f"clip_scores: {clip_scores}") 
        if score < bestscore:
            bestscore = score
            bestclip = best_clip
    clip_averages = {}
    for clip, scores in clip_scores.items():
        clip_averages[clip] = np.mean(scores)
    min_average = min(clip_averages.values())
    index = list(clip_averages.values()).index(min_average)
    min_clip_name = list(clip_averages.keys())[index]
    print(f"this took {(time.time()-starttime)/60} minutes")
    print(f'the averages for all clippings were {clip_averages}')
    print(f"the best clipping score is {bestscore} with clip: {bestclip}")

def comparing_encodings():
    """code used to compare different encodings"""
    starttime=time.time()
    print("started")
    bestscore=100
    order_of_encodings = ['ligandf', 'topologicalf', 'morganf', 'maccskeysf', 'peptidef', 'windowbasedf', 'autocorrelationf']
    encoding_bools = {'ligandf':False, 'topologicalf':True, 'morganf': True, 'maccskeysf': True, 
                      'peptidef': True, 'windowbasedf': False, 'autocorrelationf': False}
    max_features=None
    max_depth=None
    n_estimators=600
    min_samples_leaf=1
    min_samples_split=2

    data_dictionary,affinity=extract_all_features("data/train.csv",encoding_names=list(encoding_bools.keys()))
    lf_array,n_lf_features=data_dictionary['ligandf']
    tf_array,n_tf_features=data_dictionary['topologicalf']
    mo_array,n_mo_features=data_dictionary['morganf']
    ma_array,n_ma_features=data_dictionary['maccskeysf']
    pf_array,n_pf_features=data_dictionary['peptidef']
    wb_array,n_wb_features=data_dictionary['windowbasedf']
    ac_array,n_ac_features=data_dictionary['autocorrelationf']

    all_features=np.concatenate([lf_array,tf_array,mo_array,ma_array,pf_array,wb_array,ac_array],axis=1)
    n_features_list=[n_lf_features,n_tf_features,n_mo_features,n_ma_features,n_pf_features,n_wb_features,n_ac_features]
    print(f"large array has been made, time passed: {(time.time() - starttime)/60} minutes")

    data_prep_dict=make_data_prep_dict(all_features, include_only_cleaning=True, include_scaling=False, 
                                       include_clipping=False, include_PCA=False)                    #specify here what data preps you want included in the comparison
    
    true_false_combinations = create_tf_combinations(len(n_features_list), [])      #generates lists of True and False in all possible combinations with length of the number of encodings, here 7 (4 ligand + 3 protein)
    valid_tf_combinations = verify_tf_combinations(true_false_combinations)         #only returns lists that contain at least one True value for ligand encoding and one True value for protein encoding

    data_prep_scores = {}                               #for each data prepping strategy, will include a list of all mae scores that used this prepping
    for clip in list(data_prep_dict.keys()):
        data_prep_scores[clip]=[]

    all_scores = []
    for encoding_bools in valid_tf_combinations:        #encoding_bools is a list of True and False
        sliced_data_dict={}                             #will be identical to data_prep_dict but sliced to include only the encodings of this iteration
        for current_prep_name, current_X_train in data_prep_dict.items():
            sliced_X = slicing_features(current_X_train, n_features_list, encoding_bools)
            sliced_data_dict[current_prep_name] = sliced_X

        score, best_dataprep, data_prep_scores=data_prep_cv(sliced_data_dict, affinity, data_prep_scores, n_estimators, max_depth, 
                                                            min_samples_split, min_samples_leaf, max_features)                          
        if score < bestscore:
            bestscore = score
            bestbools = encoding_bools
        print(f'{encoding_bools},{score}')
   
    prep_averages = {}
    for clip, scores in data_prep_scores.items():
        prep_averages[clip] = np.mean(scores)
    min_average = min(prep_averages.values())
    index = list(prep_averages.values()).index(min_average)
    min_prep_name = list(prep_averages.keys())[index]

    print("training took "+str((time.time()-starttime)/3600)+" hours")
    print("")
    print(f"best MAE is: {bestscore}")
    print(f"and is achieved with: {bestbools}")
    print("")
    print(f"the data prep score averages are: {prep_averages}, so the best one is {min_prep_name}")


def hyperparameter_tuning():
    """code used to run the tuning of hyperparameters"""
    create_validation = False
    errors_after_tuning = False
    starttime=time.time()
    print("started tuning")
    encoding_bools = {'ligandf':False, 'topologicalf':True, 'morganf': True, 'maccskeysf': False, 
                      'peptidef': True, 'windowbasedf': False, 'autocorrelationf': False}
    uniprot_dict=extract_sequence("data/protein_info.csv")  
    smiles_train,uniprot_ids_train,y_train=data_to_SMILES_UNIProt_ID("data/train.csv")
    X_train = extract_true_features(encoding_bools, uniprot_dict, smiles_train, uniprot_ids_train)
    if create_validation is True:
        training_set, validation_set = train_validation_split(X_train,y_train,0.8)
        X_train, y_train = training_set  
        X_validation, y_validation_true = validation_set
    print(f'data array has been made, this took {(time.time()-starttime)/60} minutes')
    # scaler=set_scaling(X_train)
    # X_scaled=data_scaling(scaler,X_train)
    # print("data is scaled")
    n_estimators_grid = [400]
    max_depth_grid =[4,41,43,45]
    min_samples_split_grid = [2]
    min_samples_leaf_grid = [1]
    max_features_grid = [None]
    param_options = {'n_estimators':n_estimators_grid, 'max_depth':max_depth_grid, 'min_samples_split':min_samples_split_grid,
                     'min_samples_leaf':min_samples_leaf_grid, 'max_features':max_features_grid}
    best_params, best_score, best_estimator = hyperparams_cv(X_train,y_train,param_options,n_iter=120,cv_fold=3,search_type='grid')
    print(best_params, best_score)
    total_time = time.time()-starttime
    print(f"this took {total_time} seconds, which is {total_time/60} minutes")
    if errors_after_tuning:
        model = best_estimator
        calculate_errors(X_train=X_train, y_train_true=y_train, X_validation=X_validation, y_validation_true=y_validation_true,
                         encoding_bools=encoding_bools, rf_model=model, params=best_params)
        print(f'this took {(time.time()-starttime)/60} minutes')