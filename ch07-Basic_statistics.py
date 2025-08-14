#------------------------------------------------------------#
# R in Action (3rd ed): Chapter 7 - Python Translation      #
# Basic statistics                                           #
# requires packages: pandas, numpy, scipy, plotnine         #
# pip install pandas numpy scipy plotnine matplotlib        #
#------------------------------------------------------------#

import pandas as pd
import numpy as np
from scipy import stats
from plotnine import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

## Chapter 07
## Basic statistics

# Create mtcars-like dataset
np.random.seed(42)
mtcars_data = {
    'mpg': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2,
            17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9,
            21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4],
    'hp': [110, 110, 93, 110, 175, 105, 245, 62, 95, 123,
           123, 180, 180, 180, 205, 215, 230, 66, 52, 65,
           97, 150, 150, 245, 175, 66, 91, 113, 264, 175, 335, 109],
    'wt': [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440,
           3.440, 4.070, 3.730, 3.780, 5.250, 5.424, 5.345, 2.200, 1.615, 1.835,
           2.465, 3.520, 3.435, 3.840, 3.845, 1.935, 2.140, 1.513, 3.170, 2.770, 3.570, 2.780],
    'am': [1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
           0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}

mtcars = pd.DataFrame(mtcars_data)
print("mtcars dataset created:")
print(mtcars.head())

# Listing 7.1 Descriptive statistics with describe()
print("\n" + "="*50)
print("DESCRIPTIVE STATISTICS")
print("="*50)

myvars = ['mpg', 'hp', 'wt']
print("Basic descriptive statistics:")
print(mtcars[myvars].describe())

# Listing 7.2 Descriptive statistics via custom function
def mystats(x, na_omit=False):
    """Custom statistics function similar to R version"""
    if na_omit:
        x = x.dropna()
    
    n = len(x)
    mean_val = np.mean(x)
    std_val = np.std(x, ddof=1)
    skew_val = stats.skew(x)
    kurt_val = stats.kurtosis(x)
    
    return pd.Series({
        'n': n,
        'mean': mean_val,
        'stdev': std_val,
        'skew': skew_val,
        'kurtosis': kurt_val
    })

print("\nCustom statistics:")
result = mtcars[myvars].apply(mystats)
print(result)

# Listing 7.3-7.5 Alternative descriptive statistics
print("\nAlternative descriptive statistics methods:")

# Using scipy.stats.describe
for var in myvars:
    desc = stats.describe(mtcars[var])
    print(f"\n{var}:")
    print(f"  Count: {desc.nobs}")
    print(f"  Mean: {desc.mean:.3f}")
    print(f"  Variance: {desc.variance:.3f}")
    print(f"  Skewness: {desc.skewness:.3f}")
    print(f"  Kurtosis: {desc.kurtosis:.3f}")

# Listing 7.6 Descriptive statistics by group
print("\n" + "="*50)
print("DESCRIPTIVE STATISTICS BY GROUP")
print("="*50)

print("Statistics by transmission type (am):")
grouped_stats = mtcars.groupby('am')[myvars].apply(lambda x: x.apply(mystats))
print(grouped_stats)

# Listing 7.7 Multiple grouping variables
# Create additional grouping variable
mtcars['vs'] = np.random.choice([0, 1], len(mtcars))

print("\nStatistics by transmission (am) and engine (vs):")
multi_grouped = (mtcars.groupby(['am', 'vs'])[myvars]
                .agg(['mean', 'std', 'count'])
                .round(3))
print(multi_grouped)

# Create Salaries-like dataset for demonstration
np.random.seed(123)
salaries_data = {
    'rank': np.random.choice(['AsstProf', 'AssocProf', 'Prof'], 397),
    'sex': np.random.choice(['Male', 'Female'], 397),
    'salary': np.random.normal(100000, 30000, 397),
    'yrs.service': np.random.uniform(0, 40, 397),
    'yrs.since.phd': np.random.uniform(0, 45, 397)
}
Salaries = pd.DataFrame(salaries_data)
Salaries['salary'] = np.clip(Salaries['salary'], 50000, 200000)

print("\nSalaries dataset summary:")
salary_summary = (Salaries.groupby(['rank', 'sex'])['salary']
                 .agg(['median', 'min', 'max'])
                 .round(0))
print(salary_summary)

# Section 7.2 Frequency tables
print("\n" + "="*50)
print("FREQUENCY TABLES")
print("="*50)

# Create Arthritis-like dataset
arthritis_data = {
    'Treatment': np.random.choice(['Placebo', 'Treated'], 84),
    'Sex': np.random.choice(['Male', 'Female'], 84),
    'Improved': np.random.choice(['None', 'Some', 'Marked'], 84)
}
Arthritis = pd.DataFrame(arthritis_data)

print("Arthritis dataset:")
print(Arthritis.head())

# One way table
print("\nOne-way frequency table:")
mytable = Arthritis['Improved'].value_counts()
print("Counts:")
print(mytable)
print("\nProportions:")
print(mytable / mytable.sum())
print("\nPercentages:")
print((mytable / mytable.sum() * 100).round(1))

# Two way table
print("\nTwo-way frequency table:")
crosstab = pd.crosstab(Arthritis['Treatment'], Arthritis['Improved'])
print("Counts:")
print(crosstab)

print("\nRow proportions:")
print(crosstab.div(crosstab.sum(axis=1), axis=0).round(3))

print("\nColumn proportions:")
print(crosstab.div(crosstab.sum(axis=0), axis=1).round(3))

print("\nWith margins:")
print(pd.crosstab(Arthritis['Treatment'], Arthritis['Improved'], margins=True))

# Three-way contingency table
print("\nThree-way contingency table:")
three_way = pd.crosstab([Arthritis['Treatment'], Arthritis['Sex']], 
                       Arthritis['Improved'])
print(three_way)

# Listing 7.10 Chi-square test of independence
print("\n" + "="*50)
print("STATISTICAL TESTS")
print("="*50)

print("Chi-square test of independence:")
chi2, p_val, dof, expected = stats.chi2_contingency(crosstab)
print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_val:.4f}")
print(f"Degrees of freedom: {dof}")

# Fisher's exact test (for 2x2 tables)
if crosstab.shape == (2, 2):
    odds_ratio, fisher_p = stats.fisher_exact(crosstab)
    print(f"\nFisher's exact test p-value: {fisher_p:.4f}")
    print(f"Odds ratio: {odds_ratio:.4f}")

# Listing 7.12 Covariances and correlations
print("\n" + "="*50)
print("CORRELATIONS AND COVARIANCES")
print("="*50)

# Create states-like data
states_data = {
    'Population': np.random.uniform(100, 40000, 50),
    'Income': np.random.uniform(3000, 6500, 50),
    'Illiteracy': np.random.uniform(0.5, 2.8, 50),
    'Life Exp': np.random.uniform(68, 73, 50),
    'Murder': np.random.uniform(1, 15, 50),
    'HS Grad': np.random.uniform(40, 70, 50)
}
states = pd.DataFrame(states_data)

print("Covariance matrix:")
print(states.cov().round(2))

print("\nCorrelation matrix (Pearson):")
print(states.corr().round(3))

print("\nCorrelation matrix (Spearman):")
print(states.corr(method='spearman').round(3))

# Subset correlations
x_vars = ['Population', 'Income', 'Illiteracy', 'HS Grad']
y_vars = ['Life Exp', 'Murder']
print("\nCorrelations between X and Y variables:")
print(states[x_vars].corrwith(states[y_vars[0]]).round(3))

# Listing 7.13 Testing correlation coefficients
print("\nCorrelation significance tests:")
corr_coeff, p_value = stats.pearsonr(states['Illiteracy'], states['Life Exp'])
print(f"Illiteracy vs Life Expectancy:")
print(f"  Correlation: {corr_coeff:.4f}")
print(f"  p-value: {p_value:.4f}")

# Multiple correlation tests
from scipy.stats import pearsonr

print("\nMultiple correlation tests:")
for i, var1 in enumerate(x_vars):
    for j, var2 in enumerate(y_vars):
        corr, p_val = pearsonr(states[var1], states[var2])
        print(f"{var1} vs {var2}: r={corr:.3f}, p={p_val:.4f}")

# t-tests
print("\n" + "="*50)
print("T-TESTS")
print("="*50)

# Create UScrime-like data
uscrime_data = {
    'So': np.random.choice([0, 1], 47),  # South indicator
    'Prob': np.random.uniform(0.01, 0.1, 47),  # Probability
    'U1': np.random.uniform(10, 25, 47),  # Unemployment 1
    'U2': np.random.uniform(25, 60, 47)   # Unemployment 2
}
UScrime = pd.DataFrame(uscrime_data)

# Independent samples t-test
print("Independent samples t-test:")
south_prob = UScrime[UScrime['So'] == 1]['Prob']
non_south_prob = UScrime[UScrime['So'] == 0]['Prob']

t_stat, p_val = stats.ttest_ind(south_prob, non_south_prob)
print(f"South vs Non-South Prob:")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_val:.4f}")

print(f"  South mean: {south_prob.mean():.4f}")
print(f"  Non-South mean: {non_south_prob.mean():.4f}")

# Paired t-test
print("\nPaired t-test:")
t_stat_paired, p_val_paired = stats.ttest_rel(UScrime['U1'], UScrime['U2'])
print(f"U1 vs U2 (paired):")
print(f"  t-statistic: {t_stat_paired:.4f}")
print(f"  p-value: {p_val_paired:.4f}")

print(f"  U1 mean: {UScrime['U1'].mean():.4f}")
print(f"  U2 mean: {UScrime['U2'].mean():.4f}")

# Non-parametric tests
print("\n" + "="*50)
print("NON-PARAMETRIC TESTS")
print("="*50)

# Mann-Whitney U test
print("Mann-Whitney U test:")
u_stat, p_val_mw = stats.mannwhitneyu(south_prob, non_south_prob, 
                                     alternative='two-sided')
print(f"South vs Non-South Prob (Mann-Whitney):")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {p_val_mw:.4f}")

print(f"  South median: {south_prob.median():.4f}")
print(f"  Non-South median: {non_south_prob.median():.4f}")

# Wilcoxon signed-rank test
print("\nWilcoxon signed-rank test:")
w_stat, p_val_w = stats.wilcoxon(UScrime['U1'], UScrime['U2'])
print(f"U1 vs U2 (Wilcoxon):")
print(f"  W-statistic: {w_stat:.4f}")
print(f"  p-value: {p_val_w:.4f}")

print(f"  U1 median: {UScrime['U1'].median():.4f}")
print(f"  U2 median: {UScrime['U2'].median():.4f}")

# Kruskal-Wallis test
print("\nKruskal-Wallis test:")
# Create states with regions
states_regions = states.copy()
states_regions['region'] = np.random.choice(['Northeast', 'South', 'North Central', 'West'], 50)

groups = [states_regions[states_regions['region'] == region]['Illiteracy'].values 
          for region in states_regions['region'].unique()]

h_stat, p_val_kw = stats.kruskal(*groups)
print(f"Illiteracy by region (Kruskal-Wallis):")
print(f"  H-statistic: {h_stat:.4f}")
print(f"  p-value: {p_val_kw:.4f}")

# Group medians
print("Group medians:")
for region in states_regions['region'].unique():
    median_val = states_regions[states_regions['region'] == region]['Illiteracy'].median()
    print(f"  {region}: {median_val:.4f}")

# Visualization examples using plotnine
print("\n" + "="*50)
print("VISUALIZATION EXAMPLES")
print("="*50)

# Box plot for group comparisons
plot1 = (ggplot(states_regions, aes(x='region', y='Illiteracy')) +
         geom_boxplot(fill='lightblue') +
         labs(title='Illiteracy by Region',
              x='Region',
              y='Illiteracy Rate') +
         theme_minimal() +
         theme(axis_text_x=element_text(rotation=45, hjust=1)))
print("Box plot by region:")
print(plot1)

# Correlation heatmap using matplotlib
plt.figure(figsize=(10, 8))
correlation_matrix = states.corr()
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Scatter plot with correlation
plot2 = (ggplot(states, aes(x='Illiteracy', y='Life Exp')) +
         geom_point() +
         geom_smooth(method='lm', se=True) +
         labs(title='Life Expectancy vs Illiteracy',
              x='Illiteracy Rate',
              y='Life Expectancy') +
         theme_minimal())
print("\nScatter plot with correlation:")
print(plot2)

print("\nBasic statistics analysis complete!") 