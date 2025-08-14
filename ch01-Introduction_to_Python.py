#------------------------------------------------------------#
# R in Action (3rd ed): Chapter 1 - Python Translation      #
# Introduction to Python                                    #
# requires packages: pandas, numpy, matplotlib, plotnine   #
# pip install pandas numpy matplotlib plotnine scipy       #
#------------------------------------------------------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *
import os
import seaborn as sns

## Chapter 01
## Introduction to Python

# Listing 1.1 A sample Python session
age = [1, 3, 5, 2, 11, 9, 3, 9, 12, 3]
weight = [4.4, 5.3, 7.2, 5.2, 8.5, 7.3, 6.0, 10.4, 10.2, 6.1]

print("Mean weight:", np.mean(weight))
print("Standard deviation:", np.std(weight, ddof=1))  # ddof=1 for sample std
print("Correlation:", np.corrcoef(age, weight)[0, 1])

# Basic plot
plt.figure(figsize=(8, 6))
plt.scatter(age, weight)
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Age vs Weight')
plt.show()

# Using plotnine for ggplot2-style plotting
df = pd.DataFrame({'age': age, 'weight': weight})
plot1 = (ggplot(df, aes(x='age', y='weight')) +
         geom_point() +
         labs(x='Age', y='Weight', title='Age vs Weight') +
         theme_minimal())
print(plot1)

# Listing 1.2 An example of commands used to manage
# the Python workspace
print("Current working directory:", os.getcwd())
os.chdir(".")  # Change directory if needed

# Set display options
pd.set_option('display.precision', 3)
np.set_printoptions(precision=3)

print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)

# Listing 1.3 Working with packages and help
# Python equivalent of R package management
print("Available methods for DataFrame:")
print([method for method in dir(pd.DataFrame) if not method.startswith('_')][:10])

# Getting help
help(pd.DataFrame.describe)

# Example dataset similar to R's Arthritis
# Create a sample dataset
np.random.seed(42)
arthritis_data = {
    'Treatment': np.random.choice(['Placebo', 'Treated'], 84),
    'Sex': np.random.choice(['Male', 'Female'], 84),
    'Age': np.random.normal(50, 15, 84).astype(int),
    'Improved': np.random.choice(['None', 'Some', 'Marked'], 84)
}

arthritis = pd.DataFrame(arthritis_data)
print("Arthritis dataset:")
print(arthritis.head())
print("\nDataset info:")
print(arthritis.info())
print("\nSummary statistics:")
print(arthritis.describe())

# Example plot with the arthritis data
plot2 = (ggplot(arthritis, aes(x='Treatment', fill='Improved')) +
         geom_bar(position='dodge') +
         labs(title='Treatment Outcomes',
              x='Treatment',
              y='Count') +
         theme_minimal())
print(plot2)

print("\nPython session complete!") 