#------------------------------------------------------------#
# R in Action (3rd ed): Chapter 3 - Python Translation      #
# Basic data management                                      #
# requires packages: pandas, numpy, pandasql                #
# pip install pandas numpy pandasql                         #
#------------------------------------------------------------#

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Listing 3.1 Creating the leadership data frame
leadership = pd.DataFrame({
    'manager': [1, 2, 3, 4, 5],
    'date': ['10/24/08', '10/28/08', '10/1/08', '10/12/08', '5/1/09'],
    'country': ['US', 'US', 'UK', 'UK', 'UK'],
    'gender': ['M', 'F', 'F', 'M', 'F'],
    'age': [32, 45, 25, 39, 99],
    'q1': [5, 3, 3, 3, 2],
    'q2': [4, 5, 5, 3, 2],
    'q3': [5, 2, 5, 4, 1],
    'q4': [5, 5, 5, np.nan, 2],
    'q5': [5, 5, 2, np.nan, 1]
})

print("Leadership dataset:")
print(leadership)

# Listing 3.2 Creating new variables
# Method 1: Direct assignment
leadership['total_score'] = (leadership['q1'] + leadership['q2'] + 
                           leadership['q3'] + leadership['q4'] + leadership['q5'])

leadership['mean_score'] = (leadership['q1'] + leadership['q2'] + 
                          leadership['q3'] + leadership['q4'] + leadership['q5']) / 5

print("\nDataset with new variables:")
print(leadership)

# Method 2: Using assign method (similar to transform in R)
leadership = leadership.assign(
    total_score2 = lambda x: x.q1 + x.q2 + x.q3 + x.q4 + x.q5,
    mean_score2 = lambda x: (x.q1 + x.q2 + x.q3 + x.q4 + x.q5) / 5
)

print("\nDataset with assign method:")
print(leadership[['manager', 'total_score', 'total_score2']])

# Listing 3.3 Apply the isna() function (equivalent to is.na())
print("\nMissing values in q1-q5:")
print(leadership[['q1', 'q2', 'q3', 'q4', 'q5']].isna())

# Listing 3.4 Using dropna() to delete incomplete observations
print("\nOriginal dataset:")
print(leadership)

newdata = leadership.dropna()
print("\nDataset after removing rows with missing values:")
print(newdata)

# Listing 3.5 Converting from one data type to another
a = [1, 2, 3]
print("Original array:", a)
print("Is numeric (int):", all(isinstance(x, (int, float)) for x in a))

a = [str(x) for x in a]  # Convert to string
print("After conversion:", a)
print("Is string:", all(isinstance(x, str) for x in a))

# Using pandas for type conversion
series_a = pd.Series([1, 2, 3])
print("\nPandas Series:")
print("Original:", series_a.dtype)
series_a = series_a.astype(str)
print("After conversion:", series_a.dtype)

# Listing 3.6 Selecting observations
newdata = leadership.iloc[0:3]  # First 3 rows
print("\nFirst 3 rows:")
print(newdata)

# Conditional selection
newdata = leadership[(leadership['gender'] == 'M') & (leadership['age'] > 30)]
print("\nMale managers over 30:")
print(newdata)

# Listing 3.7 Manipulating data with pandas (equivalent to dplyr)
leadership = pd.DataFrame({
    'manager': [1, 2, 3, 4, 5],
    'date': ['10/24/08', '10/28/08', '10/1/08', '10/12/08', '5/1/09'],
    'country': ['US', 'US', 'UK', 'UK', 'UK'],
    'gender': ['M', 'F', 'F', 'M', 'F'],
    'age': [32, 45, 25, 39, 99],
    'q1': [5, 3, 3, 3, 2],
    'q2': [4, 5, 5, 3, 2],
    'q3': [5, 2, 5, 4, 1],
    'q4': [5, 5, 5, np.nan, 2],
    'q5': [5, 5, 2, np.nan, 1]
})

# Mutate: Create new columns
leadership = leadership.assign(
    total_score = lambda x: x.q1 + x.q2 + x.q3 + x.q4 + x.q5,
    mean_score = lambda x: (x.q1 + x.q2 + x.q3 + x.q4 + x.q5) / 5
)

# Recode: Replace values
leadership['gender'] = leadership['gender'].replace({'M': 'male', 'F': 'female'})

# Rename: Change column names
leadership = leadership.rename(columns={'manager': 'ID', 'gender': 'sex'})

# Arrange: Sort data
leadership = leadership.sort_values(['sex', 'total_score'])

# Select: Choose specific columns
leadership_ratings = leadership[['ID', 'mean_score']]

# Filter: Subset rows
leadership_men_high = leadership[(leadership['sex'] == 'male') & 
                                (leadership['total_score'] > 10)]

print("\nAfter pandas operations:")
print("Leadership ratings:")
print(leadership_ratings)
print("\nHigh-scoring men:")
print(leadership_men_high)

# Listing 3.8 Using SQL statements to manipulate data frames
try:
    import pandasql as ps
    
    # Load mtcars data (create a similar dataset)
    mtcars = pd.DataFrame({
        'mpg': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2],
        'cyl': [6, 6, 4, 6, 8, 6, 8, 4, 4, 6],
        'disp': [160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6],
        'hp': [110, 110, 93, 110, 175, 105, 245, 62, 95, 123],
        'drat': [3.90, 3.90, 3.85, 3.08, 3.15, 2.76, 3.21, 3.69, 3.92, 3.92],
        'wt': [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440],
        'qsec': [16.46, 17.02, 18.61, 19.44, 17.02, 20.22, 15.84, 20.00, 22.90, 18.30],
        'vs': [0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
        'am': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        'gear': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
        'carb': [4, 4, 1, 1, 2, 1, 4, 2, 2, 4]
    }, index=['Mazda RX4', 'Mazda RX4 Wag', 'Datsun 710', 'Hornet 4 Drive', 
              'Hornet Sportabout', 'Valiant', 'Duster 360', 'Merc 240D', 
              'Merc 230', 'Merc 280'])
    
    # SQL query equivalent
    query1 = "SELECT * FROM mtcars WHERE carb=1 ORDER BY mpg"
    result1 = ps.sqldf(query1)
    print("\nSQL Query Result 1:")
    print(result1)
    
    query2 = """SELECT AVG(mpg) as avg_mpg, AVG(disp) as avg_disp, gear 
                FROM mtcars 
                WHERE cyl IN (4, 6) 
                GROUP BY gear"""
    result2 = ps.sqldf(query2)
    print("\nSQL Query Result 2:")
    print(result2)
    
except ImportError:
    print("\npandasql not available. Install with: pip install pandasql")
    print("Using pandas equivalent operations:")
    
    # Pandas equivalent operations
    mtcars = pd.DataFrame({
        'mpg': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2],
        'cyl': [6, 6, 4, 6, 8, 6, 8, 4, 4, 6],
        'disp': [160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6],
        'gear': [4, 4, 4, 3, 3, 3, 3, 4, 4, 4],
        'carb': [4, 4, 1, 1, 2, 1, 4, 2, 2, 4]
    })
    
    # Query 1: Select rows where carb=1, order by mpg
    result1 = mtcars[mtcars['carb'] == 1].sort_values('mpg')
    print("\nPandas equivalent - Query 1:")
    print(result1)
    
    # Query 2: Group by gear, calculate averages for cyl in (4,6)
    result2 = (mtcars[mtcars['cyl'].isin([4, 6])]
               .groupby('gear')
               .agg({'mpg': 'mean', 'disp': 'mean'})
               .rename(columns={'mpg': 'avg_mpg', 'disp': 'avg_disp'})
               .reset_index())
    print("\nPandas equivalent - Query 2:")
    print(result2)

# Additional data management operations
print("\n" + "="*50)
print("ADDITIONAL DATA MANAGEMENT OPERATIONS")
print("="*50)

# Group operations
print("\nGroupBy operations:")
grouped = leadership.groupby('sex').agg({
    'age': ['mean', 'min', 'max'],
    'total_score': ['mean', 'std']
}).round(2)
print(grouped)

# Pivot operations
print("\nPivot table:")
pivot = leadership.pivot_table(
    values='total_score', 
    index='country', 
    columns='sex', 
    aggfunc='mean'
)
print(pivot)

# Handling missing values
print("\nMissing value handling:")
print("Count of missing values per column:")
print(leadership.isna().sum())

# Fill missing values
leadership_filled = leadership.fillna(leadership.mean())
print("\nAfter filling missing values with mean:")
print(leadership_filled.isna().sum())

print("\nBasic data management operations complete!") 