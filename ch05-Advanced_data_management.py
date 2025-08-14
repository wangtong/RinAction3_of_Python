#------------------------------------------------------------#
# R in Action (3rd ed): Chapter 5 - Python Translation      #
# Advanced data management                                   #
# requires packages: pandas, numpy, scipy, plotnine        #
# pip install pandas numpy scipy plotnine matplotlib       #
#------------------------------------------------------------#

import pandas as pd
import numpy as np
from scipy import stats
from plotnine import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

## Chapter 05
## Advanced data management

# Listing 5.1 Calculating the mean and standard deviation
x = [1, 2, 3, 4, 5, 6, 7, 8]

# Short way
print("Short way:")
print("Mean:", np.mean(x))
print("Standard deviation:", np.std(x, ddof=1))  # ddof=1 for sample std

# Long way
n = len(x)
meanx = sum(x) / n
css = sum((xi - meanx)**2 for xi in x)
sdx = np.sqrt(css / (n - 1))
print("\nLong way:")
print("Mean:", meanx)
print("Standard deviation:", sdx)

# Table 5.5 (plot normal curve)
x_norm = np.arange(-3, 3.1, 0.1)
y_norm = stats.norm.pdf(x_norm)
data_norm = pd.DataFrame({'x': x_norm, 'y': y_norm})

normal_plot = (ggplot(data_norm, aes(x='x', y='y')) +
               geom_line() +
               labs(x="Normal Deviate", y="Density") +
               scale_x_continuous(breaks=range(-3, 4)) +
               theme_minimal())
print("\nNormal curve plot:")
print(normal_plot)

# Listing 5.2 Generating pseudo-random numbers
print("\n" + "="*50)
print("RANDOM NUMBER GENERATION")
print("="*50)

# Uniform distribution
print("Random uniform numbers:")
print(np.random.uniform(0, 1, 5))
print(np.random.uniform(0, 1, 5))

# Set seed for reproducibility
np.random.seed(1234)
print("With seed 1234:")
print(np.random.uniform(0, 1, 5))

np.random.seed(1234)
print("Same seed again:")
print(np.random.uniform(0, 1, 5))

# Listing 5.3 Generating data from a multivariate normal distribution
print("\n" + "="*50)
print("MULTIVARIATE NORMAL DISTRIBUTION")
print("="*50)

np.random.seed(1234)
np.set_printoptions(precision=3)

mean = [230.7, 146.7, 3.6]
sigma = np.array([[15360.8, 6721.2, -47.1],
                  [6721.2, 4700.9, -16.5],
                  [-47.1, -16.5, 0.3]])

mydata = np.random.multivariate_normal(mean, sigma, 500)
mydata_df = pd.DataFrame(mydata, columns=['y', 'x1', 'x2'])

print("Shape:", mydata_df.shape)
print("First 10 rows:")
print(mydata_df.head(10))

# Listing 5.4 Applying functions to data objects
print("\n" + "="*50)
print("APPLYING FUNCTIONS")
print("="*50)

np.random.seed(1234)
a = 5
print("sqrt(5):", np.sqrt(a))

b = [1.243, 5.654, 2.99]
print("Original b:", b)
print("Rounded b:", [round(x) for x in b])

c = np.random.uniform(0, 1, 12).reshape(3, 4)
print("Matrix c:")
print(c)
print("log(c):")
print(np.log(c))
print("mean(c):", np.mean(c))

# Listing 5.5 Applying a function to the rows (columns) of a matrix
mydata_matrix = np.random.normal(0, 1, 30).reshape(6, 5)
print("\nMatrix:")
print(mydata_matrix)
print("Row means:")
print(np.mean(mydata_matrix, axis=1))
print("Column means:")
print(np.mean(mydata_matrix, axis=0))
print("Column means (20% trimmed):")
print([stats.trim_mean(mydata_matrix[:, i], 0.2) for i in range(mydata_matrix.shape[1])])

# Listing 5.6 A solution to the learning example
print("\n" + "="*50)
print("STUDENT GRADING EXAMPLE")
print("="*50)

students = ["John Davis", "Angela Williams", "Bullwinkle Moose",
           "David Jones", "Janice Markhammer", "Cheryl Cushing",
           "Reuven Ytzrhak", "Greg Knox", "Joel England",
           "Mary Rayburn"]
math = [502, 600, 412, 358, 495, 512, 410, 625, 573, 522]
science = [95, 99, 80, 82, 75, 85, 80, 95, 89, 86]
english = [25, 22, 18, 15, 20, 28, 15, 30, 27, 18]

roster = pd.DataFrame({
    'Student': students,
    'Math': math,
    'Science': science,
    'English': english
})

# Standardize scores
from scipy.stats import zscore
z_scores = roster[['Math', 'Science', 'English']].apply(zscore)
roster['score'] = z_scores.mean(axis=1)

# Assign grades based on quintiles
quantiles = roster['score'].quantile([0.8, 0.6, 0.4, 0.2])
roster['grade'] = 'F'
roster.loc[roster['score'] >= quantiles[0.8], 'grade'] = 'A'
roster.loc[(roster['score'] >= quantiles[0.6]) & (roster['score'] < quantiles[0.8]), 'grade'] = 'B'
roster.loc[(roster['score'] >= quantiles[0.4]) & (roster['score'] < quantiles[0.6]), 'grade'] = 'C'
roster.loc[(roster['score'] >= quantiles[0.2]) & (roster['score'] < quantiles[0.4]), 'grade'] = 'D'

# Split names
roster[['Firstname', 'Lastname']] = roster['Student'].str.split(' ', expand=True)
roster = roster[['Firstname', 'Lastname', 'Math', 'Science', 'English', 'score', 'grade']]

# Sort by lastname, firstname
roster = roster.sort_values(['Lastname', 'Firstname'])

print("Student roster with grades:")
print(roster)

# Listing 5.7 A switch example
print("\n" + "="*50)
print("SWITCH EXAMPLE")
print("="*50)

def respond_to_feeling(feeling):
    responses = {
        'happy': "I am glad you are happy",
        'afraid': "There is nothing to fear",
        'sad': "Cheer up",
        'angry': "Calm down now"
    }
    return responses.get(feeling, "I don't understand that feeling")

feelings = ["sad", "afraid"]
for feeling in feelings:
    print(respond_to_feeling(feeling))

# Listing 5.8 mystats(): a user-written function for summary statistics
print("\n" + "="*50)
print("CUSTOM STATISTICS FUNCTION")
print("="*50)

def mystats(x, parametric=True, print_stats=False):
    x = np.array(x)
    x = x[~np.isnan(x)]  # Remove NaN values
    
    if parametric:
        center = np.mean(x)
        spread = np.std(x, ddof=1)
        if print_stats:
            print(f"Mean = {center:.6f}")
            print(f"SD = {spread:.6f}")
    else:
        center = np.median(x)
        spread = stats.median_abs_deviation(x)
        if print_stats:
            print(f"Median = {center:.6f}")
            print(f"MAD = {spread:.6f}")
    
    return {'center': center, 'spread': spread}

np.random.seed(1234)
x = np.random.normal(0, 1, 500)
result = mystats(x, print_stats=True)
print("Result:", result)

# Listing 5.9 Transposing a dataset
print("\n" + "="*50)
print("TRANSPOSING DATA")
print("="*50)

# Create mtcars-like data
mtcars_data = {
    'mpg': [21.0, 21.0, 22.8, 21.4, 18.7],
    'cyl': [6, 6, 4, 6, 8],
    'disp': [160.0, 160.0, 108.0, 258.0, 360.0],
    'hp': [110, 110, 93, 110, 175]
}
cars = pd.DataFrame(mtcars_data, 
                   index=['Mazda RX4', 'Mazda RX4 Wag', 'Datsun 710', 
                         'Hornet 4 Drive', 'Hornet Sportabout'])

print("Original data:")
print(cars)
print("\nTransposed data:")
print(cars.T)

# Listing 5.10 Converting wide to long format
print("\n" + "="*50)
print("WIDE TO LONG FORMAT")
print("="*50)

data_wide = pd.DataFrame({
    'ID': ['AU', 'CN', 'PRK'],
    'Country': ['Australia', 'China', 'North Korea'],
    'LExp1990': [76.9, 69.3, 69.9],
    'LExp2000': [79.6, 72.0, 65.3],
    'LExp2010': [82.0, 75.2, 69.6]
})

print("Wide format:")
print(data_wide)

# Convert to long format
data_long = pd.melt(data_wide, 
                   id_vars=['ID', 'Country'],
                   value_vars=['LExp1990', 'LExp2000', 'LExp2010'],
                   var_name='Variable', 
                   value_name='Life_Exp')

print("\nLong format:")
print(data_long)

# Listing 5.11 Converting long to wide format
data_wide_again = data_long.pivot_table(
    index=['ID', 'Country'], 
    columns='Variable', 
    values='Life_Exp'
).reset_index()

# Clean up column names
data_wide_again.columns.name = None
print("\nBack to wide format:")
print(data_wide_again)

# Listing 5.12 & 5.13 Aggregating data
print("\n" + "="*50)
print("DATA AGGREGATION")
print("="*50)

# Extended mtcars data
mtcars_full = pd.DataFrame({
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
})

# Aggregate by cylinders and gears
aggdata = mtcars_full.groupby(['cyl', 'gear']).agg('mean').round(3)
print("Aggregated data (mean by cylinders and gears):")
print(aggdata)

# Listing 5.14 Using pandas groupby (equivalent to dplyr)
print("\n" + "="*50)
print("PANDAS GROUPBY (DPLYR EQUIVALENT)")
print("="*50)

summary_stats = (mtcars_full
                .groupby(['cyl', 'gear'])
                .agg('mean')
                .round(3))
print("Summary statistics by cylinder and gear:")
print(summary_stats)

# More complex aggregation
detailed_stats = (mtcars_full
                 .groupby(['cyl', 'gear'])
                 .agg({
                     'mpg': ['mean', 'std', 'min', 'max'],
                     'hp': ['mean', 'std'],
                     'wt': 'mean'
                 })
                 .round(3))
print("\nDetailed statistics:")
print(detailed_stats)

# Additional data management examples
print("\n" + "="*50)
print("ADDITIONAL DATA MANAGEMENT")
print("="*50)

# Creating bins/categories
mtcars_full['mpg_category'] = pd.cut(mtcars_full['mpg'], 
                                   bins=[0, 15, 20, 25, 35], 
                                   labels=['Low', 'Medium', 'High', 'Very High'])

print("MPG categories:")
print(mtcars_full[['mpg', 'mpg_category']].head(10))

# String operations
roster['name_length'] = roster['Student'].str.len() if 'Student' in roster.columns else roster['Firstname'].str.len() + roster['Lastname'].str.len() + 1
roster['lastname_upper'] = roster['Lastname'].str.upper()

print("\nString operations:")
print(roster[['Firstname', 'Lastname', 'lastname_upper']].head())

print("\nAdvanced data management complete!") 