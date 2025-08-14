"""
------------------------------------------------------------
Python in Action: Chapter 3
Basic data management
requires packages: pandas, numpy, pandasql 
------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from datetime import datetime

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

# Listing 3.2 Creating new variables
leadership['total_score'] = leadership[['q1', 'q2', 'q3', 'q4', 'q5']].sum(axis=1)
leadership['mean_score'] = leadership[['q1', 'q2', 'q3', 'q4', 'q5']].mean(axis=1)

# use assign like transform in R
leadership = leadership.assign(
    total_score=lambda x: x[['q1', 'q2', 'q3', 'q4', 'q5']].sum(axis=1),
    mean_score=lambda x: x[['q1', 'q2', 'q3', 'q4', 'q5']].mean(axis=1)
)

# Listing 3.3 Apply the is.na() function
print(leadership.iloc[:, 5:10].isna())

# Listing 3.4 Using na.omit() to delete incomplete observations
print(leadership)
newdata = leadership.dropna()
print(newdata)

# Listing 3.5 Converting from one data type to another
a = [1, 2, 3]
print(a)
print(isinstance(a, list))
a = [str(x) for x in a]
print(a)
print(isinstance(a, list))
print(all(isinstance(x, str) for x in a))

# Listing 3.6 Selecting observations
newdata = leadership.iloc[0:3, :]

newdata = leadership[(leadership['gender'] == 'M') & 
                    (leadership['age'] > 30)]

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

leadership['total_score'] = leadership[['q1', 'q2', 'q3', 'q4', 'q5']].sum(axis=1)
leadership['mean_score'] = leadership['total_score'] / 5

leadership['gender'] = leadership['gender'].map({'M': 'male', 'F': 'female'})

leadership = leadership.rename(columns={'manager': 'ID', 'gender': 'sex'})

leadership = leadership.sort_values(['sex', 'total_score'])

leadership_ratings = leadership[['ID', 'mean_score']]

leadership_men_high = leadership[(leadership['sex'] == 'male') & 
                                (leadership['total_score'] > 10)]

# Listing 3.8 Using SQL statements to manipulate data frames
import pandasql as ps

mtcars = pd.read_csv('./datasets/mtcars.csv')
newdf = ps.sqldf("""
    SELECT * FROM mtcars 
    WHERE carb=1 
    ORDER BY mpg
""")

newdf = mtcars[mtcars['carb'] == 1].sort_values('mpg')
newdf

grouped_stats = ps.sqldf("""
    SELECT 
        AVG(mpg) as avg_mpg, 
        AVG(disp) as avg_disp, 
        gear 
    FROM mtcars 
    WHERE cyl IN (4, 6) 
    GROUP BY gear
""")

grouped_stats = (mtcars[mtcars['cyl'].isin([4, 6])]
                .groupby('gear')
                .agg({'mpg': 'mean', 'disp': 'mean'})
                .reset_index()) 