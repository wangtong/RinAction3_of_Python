# ------------------------------------------------------------ #
# Python in Action (3rd ed): Chapter 5
# Advanced data management
# requires numpy, pandas, plotnine, scipy, scikit-learn
# pip install numpy pandas plotnine scipy scikit-learn
# ------------------------------------------------------------ #

import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_line, labs, scale_x_continuous
from scipy.stats import norm
from sklearn.preprocessing import scale

# Listing 5.1 Calculating the mean and standard deviation
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# short way
print(np.mean(x))
print(np.std(x, ddof=1))
# long way
n = len(x)
meanx = np.sum(x) / n
css = np.sum((x - meanx) ** 2)
sdx = np.sqrt(css / (n - 1))
print(meanx)
print(sdx)

# Table 5.5 (plot normal curve)
x_vals = np.arange(-3, 3.1, 0.1)
y_vals = norm.pdf(x_vals)
data = pd.DataFrame({'x': x_vals, 'y': y_vals})
(
    ggplot(data, aes('x', 'y')) +
    geom_line() +
    labs(x="Normal Deviate", y="Density") +
    scale_x_continuous(breaks=list(range(-3, 4)))
)

# Listing 5.2 Generating pseudo-random numbers
# from a uniform distribution
print(np.random.uniform(size=5))
print(np.random.uniform(size=5))
np.random.seed(1234)
print(np.random.uniform(size=5))
np.random.seed(1234)
print(np.random.uniform(size=5))

# Listing 5.3 Generating data from a multivariate normal distribution
mean = np.array([230.7, 146.7, 3.6])
sigma = np.array([
    [15360.8, 6721.2, -47.1],
    [6721.2, 4700.9, -16.5],
    [-47.1, -16.5, 0.3]
])
np.random.seed(1234)
mydata = np.random.multivariate_normal(mean, sigma, 500)
mydata = pd.DataFrame(mydata, columns=["y", "x1", "x2"])
print(mydata.shape)
print(mydata.head(10))

# Listing 5.4 Applying functions to data objects
np.random.seed(1234)
a = 5
print(np.sqrt(a))
b = np.array([1.243, 5.654, 2.99])
print(np.round(b))
c = np.random.uniform(size=(3, 4))
print(c)
print(np.log(c))
print(np.mean(c))

# Listing 5.5 Applying a function to the rows (columns) of a matrix
mydata = np.random.normal(size=(6, 5))
print(mydata)
print(np.mean(mydata, axis=1))  # row means
print(np.mean(mydata, axis=0))  # column means
print(np.mean(np.sort(mydata, axis=0)[1:-1], axis=0))  # trimmed mean (approximate trim=0.2)

# Listing 5.6 A solution to the learning example
Student = ["John Davis", "Angela Williams", "Bullwinkle Moose",
           "David Jones", "Janice Markhammer", "Cheryl Cushing",
           "Reuven Ytzrhak", "Greg Knox", "Joel England",
           "Mary Rayburn"]
Math = [502, 600, 412, 358, 495, 512, 410, 625, 573, 522]
Science = [95, 99, 80, 82, 75, 85, 80, 95, 89, 86]
English = [25, 22, 18, 15, 20, 28, 15, 30, 27, 18]
roster = pd.DataFrame({
    "Student": Student,
    "Math": Math,
    "Science": Science,
    "English": English
})

z = scale(roster[["Math", "Science", "English"]])
score = z.mean(axis=1)
roster["score"] = score

y = np.quantile(score, [0.8, 0.6, 0.4, 0.2])
roster["grade"] = np.select(
    [score >= y[0],
     (score < y[0]) & (score >= y[1]),
     (score < y[1]) & (score >= y[2]),
     (score < y[2]) & (score >= y[3]),
     (score < y[3])],
    ['A', 'B', 'C', 'D', 'F']
)

name_split = roster["Student"].str.split(" ", n=1, expand=True)
roster["Firstname"] = name_split[0]
roster["Lastname"] = name_split[1]
roster = roster[["Firstname", "Lastname", "Math", "Science", "English", "score", "grade"]]
roster = roster.sort_values(by=["Lastname", "Firstname"])
print(roster)

# Listing 5.7 A switch example
feelings = ["sad", "afraid"]
for i in feelings:
    print({
        "happy": "I am glad you are happy",
        "afraid": "There is nothing to fear",
        "sad": "Cheer up",
        "angry": "Calm down now"
    }.get(i, None))

# Listing 5.8 mystats(): a user-written function for summary statistics
def mystats(x, parametric=True, printout=False):
    if parametric:
        center = np.mean(x)
        spread = np.std(x, ddof=1)
    else:
        center = np.median(x)
        spread = np.median(np.abs(x - np.median(x)))
    if printout:
        if parametric:
            print(f"Mean={center}\nSD={spread}")
        else:
            print(f"Median={center}\nMAD={spread}")
    return {"center": center, "spread": spread}

np.random.seed(1234)
x = np.random.normal(size=500)
y = mystats(x)

# Listing 5.9 Transposing a dataset
cars = pd.DataFrame(
    np.array([
        [21.0, 6, 160, 110],
        [21.0, 6, 160, 110],
        [22.8, 4, 108, 93],
        [21.4, 6, 258, 110],
        [18.7, 8, 360, 175]
    ]),
    columns=["mpg", "cyl", "disp", "hp"]
)
print(cars)
print(cars.T)

# Listing 5.10 Converting a wide format data frame to a long format
data_wide = pd.DataFrame({
    "ID": ["AU", "CN", "PRK"],
    "Country": ["Australia", "China", "North Korea"],
    "LExp1990": [76.9, 69.3, 69.9],
    "LExp2000": [79.6, 72.0, 65.3],
    "LExp2010": [82.0, 75.2, 69.6]
})
print(data_wide)
data_long = pd.melt(
    data_wide,
    id_vars=["ID", "Country"],
    value_vars=["LExp1990", "LExp2000", "LExp2010"],
    var_name="Variable",
    value_name="Life_Exp"
)
print(data_long)

# Listing 5.11 Converting a long format data frame to a wide format
data_wide2 = data_long.pivot(index=["ID", "Country"], columns="Variable", values="Life_Exp").reset_index()
print(data_wide2)

# Listing 5.12 Aggregating data with the aggregate() function
from pydataset import data
mtcars = data('mtcars')
aggdata = mtcars.groupby(['cyl', 'gear']).mean(numeric_only=True).reset_index()
print(aggdata)

# Listing 5.13 Improved code for aggregating data with aggregate()
aggdata2 = mtcars.drop(columns=['cyl', 'carb']).groupby([mtcars['cyl'], mtcars['gear']]).mean(numeric_only=True).reset_index()
aggdata2.rename(columns={'cyl': 'Cylinders', 'gear': 'Gears'}, inplace=True)
print(aggdata2)

# Listing 5.14 Aggregating data with the dplyr package
aggdata3 = mtcars.groupby(['cyl', 'gear']).agg('mean').reset_index()
print(aggdata3)
