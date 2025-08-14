#-----------------------------------------------------------------------#
# Python in Action (3rd ed): Chapter 6                                       #
# Basic graphs                                                          #
# Requires: pandas, numpy, plotnine, seaborn, matplotlib, scipy         #
#-----------------------------------------------------------------------#

import pandas as pd
import numpy as np
from plotnine import *
import matplotlib.pyplot as plt
from scipy import stats

# Listing 6.1 Simple bar charts
# Arthritis dataset from vcd package in R, here we use a similar dataset
# For demonstration, we create a mock dataset
Arthritis = pd.DataFrame({
    'Improved': np.random.choice(['None', 'Some', 'Marked'], 100),
    'Treatment': np.random.choice(['Placebo', 'Treated'], 100)
})

# Simple Bar chart
(
    ggplot(Arthritis, aes(x='Improved', fill='Improved')) +
    geom_bar() +
    labs(title="Simple Bar chart", x="Improvement", y="Frequency")
).draw()
plt.show()

# Horizontal Bar chart
(
    ggplot(Arthritis, aes(x='Improved')) +
    geom_bar() +
    labs(title="Horizontal Bar chart", x="Improvement", y="Frequency") +
    coord_flip()
).draw()
plt.show()

# Listing 6.2 Stacked, grouped, and filled bar charts
(
    ggplot(Arthritis, aes(x='Treatment', fill='Improved')) +
    geom_bar(position='stack') +
    labs(title="Stacked Bar chart", x="Treatment", y="Frequency")
).draw()
plt.show()

(
    ggplot(Arthritis, aes(x='Treatment', fill='Improved')) +
    geom_bar(position='dodge') +
    labs(title="Grouped Bar chart", x="Treatment", y="Frequency")
).draw()
plt.show()

(
    ggplot(Arthritis, aes(x='Treatment', fill='Improved')) +
    geom_bar(position='fill') +
    labs(title="Filled Bar chart", x="Treatment", y="Proportion")
).draw()
plt.show()

# Listing 6.3 Bar chart for sorted mean values
from sklearn.datasets import fetch_openml
states = pd.DataFrame({
    'state.region': np.random.choice(['Northeast', 'South', 'North Central', 'West'], 50),
    'Illiteracy': np.random.rand(50) * 5
})

plotdata = states.groupby('state.region', as_index=False).agg({'Illiteracy': 'mean'})
plotdata = plotdata.rename(columns={'Illiteracy': 'mean'})

(
    ggplot(plotdata, aes(x='reorder(state.region, mean)', y='mean')) +
    geom_bar(stat='identity') +
    labs(x="Region", y="", title="Mean Illiteracy Rate")
).draw()
plt.show()

# Listing 6.4 Bar chart of mean values with error bars
plotdata = states.groupby('state.region', as_index=False).agg(
    n=('Illiteracy', 'count'),
    mean=('Illiteracy', 'mean'),
    se=('Illiteracy', lambda x: x.std()/np.sqrt(len(x)))
)

(
    ggplot(plotdata, aes(x='reorder(state.region, mean)', y='mean')) +
    geom_bar(stat='identity', fill='skyblue') +
    geom_errorbar(aes(ymin='mean-se', ymax='mean+se'), width=0.2) +
    labs(x="Region", y="", title="Mean Illiteracy Rate", subtitle="with standard error bars")
).draw()
plt.show()

# Figure 6.5 Bar chart
(
    ggplot(Arthritis, aes(x='Improved')) +
    geom_bar(fill='gold', color='black') +
    labs(title="Treatment Outcome")
).draw()
plt.show()

# Bar chart labels
mpg = pd.read_csv('./datasets/mpg.csv', index_col=0)
(
    ggplot(mpg, aes(x='model')) +
    geom_bar() +
    labs(title="Car models in the mpg dataset", y="Frequency", x="")
).draw()
plt.show()

(
    ggplot(mpg, aes(x='model')) +
    geom_bar() +
    labs(title="Car models in the mpg dataset", y="Frequency", x="") +
    coord_flip()
).draw()
plt.show()

(
    ggplot(mpg, aes(x='model')) +
    geom_bar() +
    labs(title="Model names in the mpg dataset", y="Frequency", x="") +
    theme(axis_text_x=element_text(angle=45, ha='right', size=8))
).draw()
plt.show()

# Pie charts (plotnine does not support pie charts directly, use matplotlib)
mpg_class_counts = mpg['class'].value_counts()
plt.figure()
plt.pie(mpg_class_counts, labels=mpg_class_counts.index, autopct='%1.1f%%')
plt.title("Automobiles by Car Class")
plt.show()

# Listing 6.6 Simple Tree Map (plotnine does not support treemap, use squarify)
import squarify
plotdata = mpg['manufacturer'].value_counts().reset_index()
plotdata.columns = ['manufacturer', 'n']
plt.figure(figsize=(8,6))
squarify.plot(sizes=plotdata['n'], label=plotdata['manufacturer'], alpha=.8)
plt.axis('off')
plt.title("Treemap of Car Manufacturers")
plt.show()

# Listing 6.7 Tree Map with Subgrouping (not directly supported in plotnine/squarify)
# Skipped or can be implemented with advanced matplotlib if needed

# Listing 6.8 Histograms
cars2008 = mpg[mpg['year'] == 2008]
(
    ggplot(cars2008, aes(x='hwy')) +
    geom_histogram() +
    labs(title="Default histogram")
).draw()
plt.show()

(
    ggplot(cars2008, aes(x='hwy')) +
    geom_histogram(bins=20, color="white", fill="steelblue") +
    labs(title="Colored histogram with 20 bins", x="Highway Miles Per Gallon", y="Frequency")
).draw()
plt.show()

(
    ggplot(cars2008, aes(x='hwy', y='..density..')) +
    geom_histogram(bins=20, color="white", fill="steelblue") +
    scale_y_continuous(labels=lambda l: [f'{v*100:.0f}%' for v in l]) +
    labs(title="Histogram with percentages", y="Percent", x="Highway Miles Per Gallon")
).draw()
plt.show()

(
    ggplot(cars2008, aes(x='hwy', y='..density..')) +
    geom_histogram(bins=20, color="white", fill="steelblue") +
    scale_y_continuous(labels=lambda l: [f'{v*100:.0f}%' for v in l]) +
    geom_density(color="red", size=1) +
    labs(title="Histogram with density curve", y="Percent", x="Highway Miles Per Gallon")
).draw()
plt.show()

# Listing 6.9 Kernel density plots
(
    ggplot(cars2008, aes(x='cty')) +
    geom_density() +
    labs(title="Default kernel density plot")
).draw()
plt.show()

(
    ggplot(cars2008, aes(x='cty')) +
    geom_density(fill="red") +
    labs(title="Filled kernel density plot")
).draw()
plt.show()

(
    ggplot(cars2008, aes(x='cty')) +
    geom_density(fill="red", bw=.5) +
    labs(title="Kernel density plot with bw=0.5")
).draw()
plt.show()

# Listing 6.10 Comparative kernel density plots
cars2008 = mpg[(mpg['year'] == 2008) & (mpg['cyl'] != 5)]
cars2008['Cylinders'] = cars2008['cyl'].astype(str)
(
    ggplot(cars2008, aes(x='cty', color='Cylinders', linetype='Cylinders')) +
    geom_density(size=1) +
    labs(title="Fuel Efficiecy by Number of Cylinders", x="City Miles per Gallon")
).draw()
plt.show()

(
    ggplot(cars2008, aes(x='cty', fill='Cylinders')) +
    geom_density(alpha=.4) +
    labs(title="Fuel Efficiecy by Number of Cylinders", x="City Miles per Gallon")
).draw()
plt.show()

# Box plots
from plotnine import element_text
(
    ggplot(mpg, aes(x='factor(0)', y='mpg')) +
    geom_boxplot() +
    labs(y="Miles Per Gallon", x="", title="Box Plot")
).draw()
plt.show()

cars = mpg[mpg['cyl'] != 5].copy()
cars['Cylinders'] = cars['cyl'].astype(str)
cars['Year'] = cars['year'].astype(str)
(
    ggplot(cars, aes(x='Cylinders', y='cty')) +
    geom_boxplot() +
    labs(x="Number of Cylinders", y="Miles Per Gallon", title="Car Mileage Data")
).draw()
plt.show()

(
    ggplot(cars, aes(x='Cylinders', y='cty')) +
    geom_boxplot(notch=True, fill="steelblue", varwidth=True) +
    labs(x="Number of Cylinders", y="Miles Per Gallon", title="Car Mileage Data")
).draw()
plt.show()

(
    ggplot(cars, aes(x='Cylinders', y='cty', fill='Year')) +
    geom_boxplot() +
    labs(x="Number of Cylinders", y="Miles Per Gallon", title="City Mileage by # Cylinders and Year") +
    scale_fill_manual(values=["gold", "green"])
).draw()
plt.show()

# Listing 6.11 Violin plots
mpg = sns.load_dataset('mpg').dropna(subset=['cylinders', 'cty', 'hwy', 'model', 'year'])

# Violin and box plots
cars = mpg[mpg['cylinders'] != 5].copy()
cars['Cylinders'] = cars['cylinders'].astype('category')

(
    ggplot(cars, aes(x='Cylinders', y='cty'))
    + geom_boxplot(width=0.2, fill="green")
    + geom_violin(fill="gold", alpha=0.3)
    + labs(
        x="Number of Cylinders",
        y="City Miles Per Gallon",
        title="Violin Plots of Miles Per Gallon"
    )
)

# Dot plot
plotdata = (
    mpg[mpg['year'] == 2008]
    .groupby('model', as_index=False)
    .agg(meanHwy=('hwy', 'mean'))
)
plotdata
(
    ggplot(plotdata, aes(x='meanHwy', y='model'))
    + geom_point()
    + labs(
        x="Miles Per Gallon",
        y="",
        title="Gas Mileage for Car Models"
    )
)


# Dot plot with reordered y-axis
plotdata['model_reordered'] = plotdata['model'].astype("category")
plotdata['model_reordered'] = plotdata['model_reordered'].cat.reorder_categories(
    plotdata.sort_values('meanHwy')['model']
)

(
    ggplot(plotdata, aes(x='meanHwy', y='reorder(model, meanHwy)'))
    + geom_point()
    + labs(
        x="Miles Per Gallon",
        y="",
        title="Gas Mileage for Car Models",
        subtitle="with standard error bars"
    )
)

