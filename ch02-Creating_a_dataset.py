#------------------------------------------------------------#
# R in Action (3rd ed): Chapter 2 - Python Translation      #
# Creating a dataset                                         #
# requires packages: pandas, numpy                          #
# pip install pandas numpy                                  #
#------------------------------------------------------------#

import pandas as pd
import numpy as np

## Chapter 02
## Creating a dataset

# Types of arrays/lists (equivalent to R vectors)
a = [1, 2, 5, 3, 6, -2, 4]  # numeric list
b = ["one", "two", "three"]  # string list
c = [True, True, True, False, True, False]  # boolean list

print("Numeric array:", a)
print("String array:", b)
print("Boolean array:", c)

# Converting to numpy arrays
a_array = np.array(a)
b_array = np.array(b)
c_array = np.array(c)

print("\nNumPy arrays:")
print("Numeric:", a_array)
print("String:", b_array)
print("Boolean:", c_array)

# Subsetting arrays (equivalent to R vector subsetting)
a = ["k", "j", "h", "a", "c", "m"]
print("\nOriginal array:", a)
print("Third element (index 2):", a[2])  # Python uses 0-based indexing
print("Elements 1, 3, 5 (indices 0, 2, 4):", [a[i] for i in [0, 2, 4]])
print("Elements 2-6 (indices 1-5):", a[1:6])

# Listing 2.1 Creating matrices (using numpy)
y = np.arange(1, 21).reshape(5, 4)  # 5 rows, 4 columns
print("\nMatrix y:")
print(y)

cells = [1, 26, 24, 68]
mymatrix = np.array(cells).reshape(2, 2)  # 2x2 matrix, row-wise (default)
print("\nMatrix by rows:")
print(mymatrix)

# Create matrix with column names and row names using pandas DataFrame
rnames = ["R1", "R2"]
cnames = ["C1", "C2"]
mymatrix_df = pd.DataFrame(mymatrix, index=rnames, columns=cnames)
print("\nMatrix with names:")
print(mymatrix_df)

# Column-wise filling
mymatrix_col = np.array(cells).reshape(2, 2, order='F')  # Fortran order (column-wise)
mymatrix_col_df = pd.DataFrame(mymatrix_col, index=rnames, columns=cnames)
print("\nMatrix by columns:")
print(mymatrix_col_df)

# Listing 2.2 Using matrix subscripts
x = np.arange(1, 11).reshape(2, 5)  # 2 rows, 5 columns
print("\nMatrix x:")
print(x)
print("Second row:", x[1, :])          # Second row (index 1)
print("Second column:", x[:, 1])       # Second column (index 1)
print("Element (1,4):", x[0, 3])       # First row, fourth column (0-indexed)
print("Elements (1,4) and (1,5):", x[0, [3, 4]])  # First row, 4th and 5th columns

# Listing 2.3 Creating an array (3D array)
dim1 = ["A1", "A2"]
dim2 = ["B1", "B2", "B3"]
dim3 = ["C1", "C2", "C3", "C4"]

# Create 3D array: 2x3x4
z = np.arange(1, 25).reshape(2, 3, 4)
print("\n3D Array z (2x3x4):")
print(z)

# Using xarray for labeled dimensions (more similar to R arrays)
try:
    import xarray as xr
    z_xr = xr.DataArray(
        z,
        dims=['dim1', 'dim2', 'dim3'],
        coords={'dim1': dim1, 'dim2': dim2, 'dim3': dim3}
    )
    print("\nLabeled 3D array:")
    print(z_xr)
except ImportError:
    print("\nxarray not available. Install with: pip install xarray")

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
print("\nPatient data frame:")
print(patientdata)

# Listing 2.5 Accessing elements of a data frame
print("\nFirst two columns:")
print(patientdata.iloc[:, 0:2])  # First two columns

print("\nAge column:")
print(patientdata['age'])  # Access by column name

print("\nDiabetes and status columns:")
print(patientdata[['diabetes', 'status']])  # Multiple columns

print("\nAge column (alternative access):")
print(patientdata.age)  # Dot notation

# Listing 2.6 Using categorical data (factors)
patientID = [1, 2, 3, 4]
age = [25, 34, 28, 52]
diabetes = ["Type1", "Type2", "Type1", "Type1"]
status = ["Poor", "Improved", "Excellent", "Poor"]

# Convert to categorical data
diabetes_cat = pd.Categorical(diabetes)
status_cat = pd.Categorical(status, categories=["Poor", "Improved", "Excellent"], ordered=True)

patientdata = pd.DataFrame({
    'patientID': patientID,
    'age': age,
    'diabetes': diabetes_cat,
    'status': status_cat
})

print("\nPatient data with categorical variables:")
print(patientdata.dtypes)
print("\nData info:")
print(patientdata.info())
print("\nSummary:")
print(patientdata.describe(include='all'))

# Listing 2.7 Creating a list-like structure (dictionary)
g = "My First Dictionary"
h = [25, 26, 18, 39]
j = np.arange(1, 11).reshape(5, 2)
k = ["one", "two", "three"]

# Python dictionary (similar to R list)
mydict = {
    'title': g,
    'ages': h,
    'matrix': j,
    'strings': k
}

print("\nPython dictionary:")
for key, value in mydict.items():
    print(f"{key}:")
    print(value)
    print()

# Accessing dictionary elements
print("Ages:", mydict['ages'])
print("Second element of ages:", mydict['ages'][1])

# Alternative: using pandas Series for more structure
myseries = pd.Series([g, h, j, k], index=['title', 'ages', 'matrix', 'strings'])
print("\nPandas Series structure:")
print(myseries)

print("\nData structures creation complete!") 