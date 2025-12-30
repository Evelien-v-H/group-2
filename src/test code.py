import numpy as np
import matplotlib as plt
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
                if already_errorvalue is False:
                    values_colom=[]
                    for k in range(data.shape[0]):
                        if is_number(data[k,i]) is True:
                            values_colom.append(float(data[k,i]))
                            print(values_colom)
                    
                    print(len(values_colom))
                    if len(values_colom)!=0:
                        mean_value_colom=np.mean(values_colom)
                        data[j,i]=float(mean_value_colom)
                    
                    else:
                        irrelevant_colums.append(i)
                    already_errorvalue=True
                    
            
                else:
                    if i not in irrelevant_colums:
                        data[j,i]=float(mean_value_colom)
    print(data)
    print(irrelevant_colums)
    if len(irrelevant_colums)>0:
        irrelevant_colums.reverse()
        for n in irrelevant_colums:
            data = np.delete(data, n, axis=1)

    return data

array=[1,2,3]
array1=[4,5,6]
array3=[7,8,9]
matrix=np.vstack((array,array1))
matrix=np.vstack((matrix,array3))

for i in range(3):
    print(matrix[i:i+1,:])
    print('/n')

    #Voor het overzichtelijk maken van de data
def boxplot_everything(matrix):
    """Making a boxplot of every kolom
    
    input: matrix, preferably scaled
    
    output: nothing"""
    #Ik kan geen labels toevoegen want weet niet wat alles is en hoe alles is opgebouwd, dus deze functie schiet niet zoveel op, 
    # maar is ook deels voor overzicht bedoeld
    plt.boxplot(matrix)
    plt.show()

print('a')
data_dictionary,affinity=extract_all_features("data/train.csv")
print('b')
plt.hist(affinity)
plt.show()
print('*')
lf_array,lf_features=data_dictionary['ligandf']
tf_array,tf_features=data_dictionary['topologicalf']
mo_array,mo_features=data_dictionary['morganf']
ma_array,ma_features=data_dictionary['macckeysf']
pf_array,pf_features=data_dictionary['peptidef']
wb_array,wb_features=data_dictionary['windowbasedf']
ac_array,ac_features=data_dictionary['autocorrelationf']
print('c')
all_features=np.concatenate([lf_array,tf_array,mo_array,ma_array,pf_array,wb_array,ac_array],axis=1)
print('d')
n_features_list=[lf_features,tf_features,mo_features,ma_features,pf_features,wb_features,ac_features]
order_of_encodings = ['ligandf', 'topological', 'morgan', 'macckeys', 'peptidef', 'windowbased', 'autocorrelation']

encoding_bools=[True,True,True,True,True,True,True]
X = slicing_features(all_features, n_features_list, encoding_bools)
y = affinity
print('e')
X_clipped,list=clipping_outliers_train(X)
scaler=set_scaling(X_clipped)
X_scaled=data_scaling(scaler,X_clipped)
print('f')
boxplot_everything(X_scaled)
print('g')

