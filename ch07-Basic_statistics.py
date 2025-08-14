#------------------------------------------------------------#
# Python in Action (3rd ed): Chapter 7                            #
# Basic statistics                                           #
# Requires: pandas, numpy, scipy, plotnine                   #
#------------------------------------------------------------#

import pandas as pd
import numpy as np
from scipy import stats
from plotnine import *

# Listing 7.1 Descriptive statistics with summary()
mtcars = pd.read_csv('./datasets/mtcars.csv', index_col=0)
myvars = ["mpg", "hp", "wt"]
print(mtcars[myvars].describe())

# Listing 7.2 Descriptive statistics via sapply()
def mystats(x, na_omit=False):
    if na_omit:
        x = x.dropna()
    m = x.mean()
    n = x.count()
    s = x.std()
    skew = ((x - m) ** 3).sum() / (s ** 3) / n
    kurt = ((x - m) ** 4).sum() / (s ** 4) / n - 3
    return pd.Series({'n': n, 'mean': m, 'stdev': s, 'skew': skew, 'kurtosis': kurt})

print(mtcars[myvars].apply(mystats))

# Listing 7.3-7.5: Descriptive statistics via Hmisc, pastecs, psych (use pandas)
print(mtcars[myvars].describe(include='all'))

# Listing 7.6 Descriptive statistics by group using by()
def dstats(df):
    return df.apply(mystats)
print(mtcars.groupby('am')[myvars].apply(dstats))

# Listing 7.7 Descriptive statistics for groups defined by multiple variables
print(mtcars.groupby(['am', 'vs'])[myvars].apply(lambda df: df.apply(lambda x: mystats(x, na_omit=True))))

# Section 7.1.4 Summarizing data interactively with dplyr
# Salaries dataset from carData in R, here we use a mock dataset
Salaries = pd.DataFrame({
    'salary': np.random.normal(80000, 15000, 200),
    'rank': np.random.choice(['AsstProf', 'AssocProf', 'Prof'], 200),
    'sex': np.random.choice(['Male', 'Female'], 200),
    'yrs.service': np.random.randint(1, 40, 200),
    'yrs.since.phd': np.random.randint(1, 40, 200)
})
print(Salaries['salary'].median(), Salaries['salary'].min(), Salaries['salary'].max())
print(Salaries.groupby(['rank', 'sex'])['salary'].agg(['count', 'median', 'min', 'max']))
print(Salaries.groupby(['rank', 'sex'])[['yrs.service', 'yrs.since.phd']].mean())

# Section 7.2 Frequency tables
# Arthritis dataset from vcd package in R, here we use a similar dataset
Arthritis = pd.read_csv('data/Arthritis.csv',index_col=0)
print(Arthritis['Improved'].value_counts())
print(Arthritis['Improved'].value_counts(normalize=True))
print(Arthritis['Improved'].value_counts(normalize=True) * 100)

mytable = pd.crosstab(Arthritis['Treatment'], Arthritis['Improved'])
print(mytable)
print(mytable.sum(axis=1))
print(mytable.div(mytable.sum(axis=1), axis=0))
print(mytable.sum(axis=0))
print(mytable.div(mytable.sum(axis=0), axis=1))
print(mytable / mytable.values.sum())
print(mytable.append(mytable.sum(axis=0), ignore_index=True))

# Listing 7.8 Two-way table using CrossTable (use pandas crosstab)
print(pd.crosstab(Arthritis['Treatment'], Arthritis['Improved'], margins=True, normalize=False))

# Listing 7.9 Three-way contingency table
mytable3 = pd.crosstab([Arthritis['Treatment'], Arthritis['Sex']], Arthritis['Improved'])
print(mytable3)

# Listing 7.10 Chi-square test of independence
chi2, p, dof, expected = stats.chi2_contingency(mytable)
print('Chi-square:', chi2, 'p-value:', p)

# Fisher's exact test (only for 2x2 tables)
if mytable.shape == (2,2):
    oddsr, p = stats.fisher_exact(mytable)
    print('Fisher exact:', oddsr, 'p-value:', p)

# Listing 7.11 Measures of association for a two-way table (Cramer's V)
def cramers_v(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    return np.sqrt(phi2/min(k-1, r-1))
print('Cramer\'s V:', cramers_v(mytable))

# Listing 7.12 Covariances and correlations
states = pd.DataFrame(np.random.rand(50,6), columns=["Population", "Income", "Illiteracy", "HS Grad", "Life Exp", "Murder"])
print(states.cov())
print(states.corr())
print(states.corr(method="spearman"))

x = states[["Population", "Income", "Illiteracy", "HS Grad"]]
y = states[["Life Exp", "Murder"]]
print(x.corrwith(y["Life Exp"]))
print(x.corrwith(y["Murder"]))

# partial correlations (not directly in scipy, can use pingouin or manual)
# Listing 7.13 Testing a correlation coefficient for significance
corr_coef, p_value = stats.pearsonr(states.iloc[:, 2], states.iloc[:, 4])
print("Correlation coefficient:", corr_coef, "p-value:", p_value)

# Listing 7.14 Correlation matrix and tests of significance via corr.test()
corr_matrix = states.corr()
corr_test = pg.pairwise_corr(states, method='pearson')
print(corr_matrix)
print(corr_test)

# t-tests
t_stat, t_p = stats.ttest_ind(UScrime.loc[UScrime['So'] == 0, 'Prob'],
                              UScrime.loc[UScrime['So'] == 1, 'Prob'])
print("t-test:", t_stat, "p-value:", t_p)

means_sds = UScrime[["U1", "U2"]].agg(['mean', 'std'])
print(means_sds)

with_paired_t_stat, with_paired_t_p = stats.ttest_rel(UScrime["U1"], UScrime["U2"])
print("Paired t-test:", with_paired_t_stat, "p-value:", with_paired_t_p)

# Mann-Whitney U-test
medians = UScrime.groupby('So')['Prob'].median()
print(medians)
u_stat, u_p = stats.mannwhitneyu(UScrime.loc[UScrime['So'] == 0, 'Prob'],
                                 UScrime.loc[UScrime['So'] == 1, 'Prob'],
                                 alternative='two-sided')
print("Mann-Whitney U-test:", u_stat, "p-value:", u_p)

medians_u = UScrime[["U1", "U2"]].median()
print(medians_u)

paired_u_stat, paired_u_p = stats.wilcoxon(UScrime["U1"], UScrime["U2"])
print("Paired Wilcoxon test:", paired_u_stat, "p-value:", paired_u_p)

# Kruskal-Wallis test
states_df = pd.DataFrame({'state_region': state_region, **state_x77})
kruskal_stat, kruskal_p = stats.kruskal(
    *[group["Illiteracy"].values for name, group in states_df.groupby("state_region")]
)
print("Kruskal-Wallis test:", kruskal_stat, "p-value:", kruskal_p)

# Listing 7.15 Nonparametric multiple comparisons
dunn_result = sp.posthoc_dunn(states_df, val_col='Illiteracy', group_col='state_region', p_adjust='holm')
print(dunn_result)
