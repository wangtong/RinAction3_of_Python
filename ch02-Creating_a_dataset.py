"""
------------------------------------------------------------
Python in Action: Chapter 2
Creating a dataset
------------------------------------------------------------
"""

import numpy as np
import pandas as pd

# Types of nd.array
a = np.array([1, 2, 5, 3, 6, -2, 4])  # numeric
b = np.array(["one", "two", "three"])  # character
c = np.array([True, True, True, False, True, False])  # logical

# subsetting vectors
a = np.array(["k", "j", "h", "a", "c", "m"])
a[2]
a[[0, 2, 4]]
a[1:6]

# Listing 2.1 Creating matrices
y = np.array(range(1, 21)).reshape(5, 4)
y

cells = [1, 26, 24, 68]
rnames = ["R1", "R2"]
cnames = ["C1", "C2"]
mymatrix = np.array(cells).reshape(2, 2)
mymatrix = pd.DataFrame(mymatrix, index=rnames, columns=cnames)
print(mymatrix)

# Listing 2.2 Using matrix subscripts
x = np.array(range(1, 11)).reshape(2, 5)
print(x)
print(x[1, :]) 
print(x[:, 1])
print(x[0, 3])  
print(x[0, [3, 4]])  

# Listing 2.3 Creating an array
dim1 = ["A1", "A2"]
dim2 = ["B1", "B2", "B3"]
dim3 = ["C1", "C2", "C3", "C4"]
z = np.array(range(1, 25)).reshape(2, 3, 4)
z_dict = {(i, j, k): z[i-1, j-1, k-1] 
          for i, d1 in enumerate(dim1, 1)
          for j, d2 in enumerate(dim2, 1)
          for k, d3 in enumerate(dim3, 1)}
z

# Listing 2.4 Creating a data frame
patientID = [1, 2, 3, 4]
age = [25, 34, 28, 52]
diabetes = ["Type1", "Type2", "Type1", "Type1"]
status = ["Poor", "Improved", "Excellent", "Poor"]
patientdata = pd.DataFrame({
    'patientID': patientID,
    'age': age,
    'diabetes': diabetes,
    'status': status
})
print(patientdata)

# Listing 2.5 Specifying elements of a data frame
print(patientdata.iloc[:, 0:2]) 
print(patientdata['age'])
print(patientdata[['diabetes', 'status']])
print(patientdata.loc[: ,'age'])

# Listing 2.6 Using factors (categorical data in pandas)
patientID = [1, 2, 3, 4]
age = [25, 34, 28, 52]
diabetes = ["Type1", "Type2", "Type1", "Type1"]
status = ["Poor", "Improved", "Excellent", "Poor"]
patientdata['diabetes'] = pd.Categorical(patientdata['diabetes'])
patientdata['status'] = pd.Categorical(patientdata['status'], 
                                     categories=['Poor', 'Improved', 'Excellent'],
                                     ordered=True)
print(str(patientdata))
print(patientdata.dtypes)
print(patientdata.describe())

# Listing 2.7 Creating a list (dictionary in Python)
g = "My First List"
h = [25, 26, 18, 39]
j = np.array(range(1, 11)).reshape(5, 2)
k = ["one", "two", "three"]
mylist = {
    'title': g,
    'ages': h,
    'matrix': j,
    'strings': k
}
print(mylist)
print(mylist['matrix'][1])
print(mylist['ages']) 
