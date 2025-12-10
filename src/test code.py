import numpy as np
array1=np.array([1,2,3])
array2=np.array([4,5,6])
array3=np.array([7,8,9])

matrix=np.vstack([array1,array2])
print(matrix)
matrix=np.vstack([matrix,array3])
print(matrix)
