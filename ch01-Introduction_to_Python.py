"""
------------------------------------------------------------
Python in Action: Chapter 1
Introduction to Python
------------------------------------------------------------
"""

## Chapter 01
# Listing 1.1 A sample Python session
import numpy as np
import matplotlib.pyplot as plt

age = np.array([1, 3, 5, 2, 11, 9, 3, 9, 12, 3])
weight = np.array([4.4, 5.3, 7.2, 5.2, 8.5, 7.3, 6.0, 10.4, 10.2, 6.1])

np.mean(weight)
np.std(weight)
np.corrcoef(age, weight)[0,1]

# Plot Scatterplot
plt.scatter(age, weight)
plt.xlabel('Age')
plt.ylabel('Weight')
plt.title('Age vs Weight')
plt.show()

# Listing 1.2 An example of commands used to manage
# the Python workspace
import os
os.chdir("C:/myprojects/project1")
np.set_printoptions(precision=3)

# Listing 1.3 Working with a new package
# in Python，we use pip to install modules
# !pip install statsmodels

import statsmodels.api as sm
from statsmodels.datasets import get_rdataset
arthritis = get_rdataset("Arthritis", "vcd")
arthritis.data.head()
arthritis.__doc__