#-----------------------------------------------------------------------#
# R in Action (3rd ed): Chapter 6 - Python Translation                 #
# Basic graphs                                                          #
# requires packages: plotnine, pandas, matplotlib, seaborn             #
# pip install plotnine pandas matplotlib seaborn squarify              #
#-----------------------------------------------------------------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

## Chapter 06
## Basic graphs

# Create sample datasets
# Load mpg-like data
np.random.seed(42)
n = 234

# Create mpg dataset (similar to ggplot2::mpg)
mpg_data = {
    'manufacturer': np.random.choice(['audi', 'chevrolet', 'dodge', 'ford', 'honda', 
                                     'hyundai', 'jeep', 'land rover', 'lincoln', 
                                     'mercury', 'nissan', 'pontiac', 'subaru', 
                                     'toyota', 'volkswagen'], n),
    'model': np.random.choice(['a4', 'altima', 'c1500', 'caravan', 'civic', 
                              'corolla', 'explorer', 'f150', 'grand prix', 
                              'impreza', 'jetta', 'passat', 'ranger', 
                              'sonata', 'tiburon'], n),
    'displ': np.random.uniform(1.0, 7.0, n),
    'year': np.random.choice([1999, 2008], n),
    'cyl': np.random.choice([4, 5, 6, 8], n, p=[0.4, 0.1, 0.3, 0.2]),
    'trans': np.random.choice(['auto', 'manual'], n, p=[0.7, 0.3]),
    'drv': np.random.choice(['4', 'f', 'r'], n, p=[0.3, 0.5, 0.2]),
    'cty': np.random.randint(9, 35, n),
    'hwy': np.random.randint(12, 44, n),
    'fl': np.random.choice(['c', 'd', 'e', 'p', 'r'], n),
    'class': np.random.choice(['2seater', 'compact', 'midsize', 'minivan', 
                              'pickup', 'subcompact', 'suv'], n)
}

mpg = pd.DataFrame(mpg_data)
mpg['hwy'] = mpg['cty'] + np.random.randint(2, 10, n)  # Highway usually higher than city

# Create Arthritis dataset
arthritis_data = {
    'Treatment': np.random.choice(['Placebo', 'Treated'], 84),
    'Sex': np.random.choice(['Male', 'Female'], 84),
    'Age': np.random.normal(50, 15, 84).astype(int),
    'Improved': np.random.choice(['None', 'Some', 'Marked'], 84)
}
Arthritis = pd.DataFrame(arthritis_data)

print("Datasets created successfully!")
print(f"MPG dataset shape: {mpg.shape}")
print(f"Arthritis dataset shape: {Arthritis.shape}")

# Listing 6.1 Simple bar charts
print("\n" + "="*50)
print("BAR CHARTS")
print("="*50)

plot1 = (ggplot(Arthritis, aes(x='Improved', fill='Improved')) + 
         geom_bar() +
         labs(title="Simple Bar chart",
              x="Improvement", 
              y="Frequency") +
         theme_minimal())
print("Simple bar chart:")
print(plot1)

plot2 = (ggplot(Arthritis, aes(x='Improved')) + 
         geom_bar() +
         labs(title="Horizontal Bar chart",
              x="Improvement",
              y="Frequency") +
         coord_flip() +
         theme_minimal())
print("\nHorizontal bar chart:")
print(plot2)

# Listing 6.2 Stacked, grouped, and filled bar charts
plot3 = (ggplot(Arthritis, aes(x='Treatment', fill='Improved')) +
         geom_bar(position='stack') +
         labs(title="Stacked Bar chart",
              x="Treatment",
              y="Frequency") +
         theme_minimal())
print("\nStacked bar chart:")
print(plot3)

plot4 = (ggplot(Arthritis, aes(x='Treatment', fill='Improved')) +
         geom_bar(position='dodge') +
         labs(title="Grouped Bar chart",
              x="Treatment",
              y="Frequency") +
         theme_minimal())
print("\nGrouped bar chart:")
print(plot4)

plot5 = (ggplot(Arthritis, aes(x='Treatment', fill='Improved')) +
         geom_bar(position='fill') +
         labs(title="Filled Bar chart",
              x="Treatment",
              y="Proportion") +
         theme_minimal())
print("\nFilled bar chart:")
print(plot5)

# Listing 6.3 Bar chart for sorted mean values
# Create states-like data
states_data = {
    'state.region': np.repeat(['Northeast', 'South', 'North Central', 'West'], [9, 16, 12, 13]),
    'Illiteracy': np.random.uniform(0.5, 2.8, 50)
}
states = pd.DataFrame(states_data)

plotdata = (states.groupby('state.region')['Illiteracy']
           .mean()
           .reset_index()
           .rename(columns={'Illiteracy': 'mean'}))

plot6 = (ggplot(plotdata, aes(x='reorder(state_region, mean)', y='mean')) +
         geom_bar(stat='identity') +
         labs(x="Region", y="", title="Mean Illiteracy Rate") +
         theme_minimal() +
         theme(axis_text_x=element_text(rotation=45, hjust=1)))
print("\nSorted bar chart:")
print(plot6)

# Listing 6.4 Bar chart with error bars
plotdata_se = (states.groupby('state.region')['Illiteracy']
              .agg(['count', 'mean', 'std'])
              .reset_index())
plotdata_se['se'] = plotdata_se['std'] / np.sqrt(plotdata_se['count'])

plot7 = (ggplot(plotdata_se, aes(x='reorder(state_region, mean)', y='mean')) +
         geom_bar(stat='identity', fill='skyblue') +
         geom_errorbar(aes(ymin='mean-se', ymax='mean+se'), width=0.2) +
         labs(x="Region", y="", title="Mean Illiteracy Rate",
              subtitle="with standard error bars") +
         theme_minimal() +
         theme(axis_text_x=element_text(rotation=45, hjust=1)))
print("\nBar chart with error bars:")
print(plot7)

# Pie charts (using matplotlib since plotnine doesn't have native pie charts)
print("\n" + "="*50)
print("PIE CHARTS")
print("="*50)

class_counts = mpg['class'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%')
plt.title('Automobiles by Car Class')
plt.axis('equal')
plt.show()

print("Pie chart created with matplotlib")

# Tree maps (using matplotlib and custom implementation)
print("\n" + "="*50)
print("TREE MAPS")
print("="*50)

try:
    import squarify
    
    # Simple tree map
    plotdata_tree = mpg['manufacturer'].value_counts().head(10)
    
    plt.figure(figsize=(12, 8))
    squarify.plot(sizes=plotdata_tree.values, 
                 label=plotdata_tree.index,
                 alpha=0.8)
    plt.title('Car Manufacturers')
    plt.axis('off')
    plt.show()
    
    print("Tree map created with squarify")
    
except ImportError:
    print("squarify not available. Install with: pip install squarify")

# Listing 6.8 Histograms
print("\n" + "="*50)
print("HISTOGRAMS")
print("="*50)

cars2008 = mpg[mpg['year'] == 2008]

plot8 = (ggplot(cars2008, aes(x='hwy')) +
         geom_histogram() +
         labs(title="Default histogram") +
         theme_minimal())
print("Default histogram:")
print(plot8)

plot9 = (ggplot(cars2008, aes(x='hwy')) +
         geom_histogram(bins=20, color='white', fill='steelblue') +
         labs(title="Colored histogram with 20 bins",
              x="Highway Miles Per Gallon",
              y="Frequency") +
         theme_minimal())
print("\nColored histogram:")
print(plot9)

plot10 = (ggplot(cars2008, aes(x='hwy', y='..density..')) +
          geom_histogram(bins=20, color='white', fill='steelblue') +
          labs(title="Histogram with percentages",
               y="Percent",
               x="Highway Miles Per Gallon") +
          theme_minimal())
print("\nHistogram with density:")
print(plot10)

plot11 = (ggplot(cars2008, aes(x='hwy', y='..density..')) +
          geom_histogram(bins=20, color='white', fill='steelblue') +
          geom_density(color='red', size=1) +
          labs(title="Histogram with density curve",
               y="Percent",
               x="Highway Miles Per Gallon") +
          theme_minimal())
print("\nHistogram with density curve:")
print(plot11)

# Listing 6.9 Kernel density plots
print("\n" + "="*50)
print("DENSITY PLOTS")
print("="*50)

plot12 = (ggplot(cars2008, aes(x='cty')) +
          geom_density() +
          labs(title="Default kernel density plot") +
          theme_minimal())
print("Default density plot:")
print(plot12)

plot13 = (ggplot(cars2008, aes(x='cty')) +
          geom_density(fill='red') +
          labs(title="Filled kernel density plot") +
          theme_minimal())
print("\nFilled density plot:")
print(plot13)

# Comparative density plots
cars_no5 = mpg[(mpg['year'] == 2008) & (mpg['cyl'] != 5)]
cars_no5['Cylinders'] = cars_no5['cyl'].astype(str)

plot14 = (ggplot(cars_no5, aes(x='cty', color='Cylinders', linetype='Cylinders')) +
          geom_density(size=1) +
          labs(title="Fuel Efficiency by Number of Cylinders",
               x="City Miles per Gallon") +
          theme_minimal())
print("\nComparative density plot (lines):")
print(plot14)

plot15 = (ggplot(cars_no5, aes(x='cty', fill='Cylinders')) +
          geom_density(alpha=0.4) +
          labs(title="Fuel Efficiency by Number of Cylinders",
               x="City Miles per Gallon") +
          theme_minimal())
print("\nComparative density plot (filled):")
print(plot15)

# Box plots
print("\n" + "="*50)
print("BOX PLOTS")
print("="*50)

# Simple box plot using mtcars-like data
mtcars_sample = pd.DataFrame({
    'mpg': np.random.normal(20, 6, 32),
    'cyl': np.random.choice([4, 6, 8], 32),
    'hp': np.random.normal(150, 50, 32)
})

plot16 = (ggplot(mtcars_sample, aes(x='""', y='mpg')) +
          geom_boxplot() +
          labs(y="Miles Per Gallon", x="", title="Box Plot") +
          theme_minimal())
print("Simple box plot:")
print(plot16)

cars_cyl = mpg[mpg['cyl'] != 5].copy()
cars_cyl['Cylinders'] = cars_cyl['cyl'].astype(str)
cars_cyl['Year'] = cars_cyl['year'].astype(str)

plot17 = (ggplot(cars_cyl, aes(x='Cylinders', y='cty')) +
          geom_boxplot() +
          labs(x="Number of Cylinders",
               y="Miles Per Gallon",
               title="Car Mileage Data") +
          theme_minimal())
print("\nBox plot by cylinders:")
print(plot17)

plot18 = (ggplot(cars_cyl, aes(x='Cylinders', y='cty')) +
          geom_boxplot(notch=True, fill='steelblue') +
          labs(x="Number of Cylinders",
               y="Miles Per Gallon",
               title="Car Mileage Data") +
          theme_minimal())
print("\nNotched box plot:")
print(plot18)

plot19 = (ggplot(cars_cyl, aes(x='Cylinders', y='cty', fill='Year')) +
          geom_boxplot() +
          labs(x="Number of Cylinders",
               y="Miles Per Gallon",
               title="City Mileage by # Cylinders and Year") +
          scale_fill_manual(values=['gold', 'green']) +
          theme_minimal())
print("\nGrouped box plot:")
print(plot19)

# Listing 6.11 Violin plots
print("\n" + "="*50)
print("VIOLIN PLOTS")
print("="*50)

plot20 = (ggplot(cars_cyl, aes(x='Cylinders', y='cty')) +
          geom_boxplot(width=0.2, fill='green') +
          geom_violin(fill='gold', alpha=0.3) +
          labs(x="Number of Cylinders",
               y="City Miles Per Gallon",
               title="Violin Plots of Miles Per Gallon") +
          theme_minimal())
print("Violin plot with box plot:")
print(plot20)

# Dot plots
print("\n" + "="*50)
print("DOT PLOTS")
print("="*50)

plotdata_dot = (mpg[mpg['year'] == 2008]
                .groupby('model')['hwy']
                .mean()
                .reset_index()
                .rename(columns={'hwy': 'meanHwy'}))

plot21 = (ggplot(plotdata_dot, aes(x='meanHwy', y='model')) +
          geom_point() +
          labs(x="Miles Per Gallon",
               y="",
               title="Gas Mileage for Car Models") +
          theme_minimal())
print("Simple dot plot:")
print(plot21)

plot22 = (ggplot(plotdata_dot, aes(x='meanHwy', y='reorder(model, meanHwy)')) +
          geom_point() +
          labs(x="Miles Per Gallon",
               y="",
               title="Gas Mileage for Car Models",
               subtitle="sorted by mileage") +
          theme_minimal())
print("\nSorted dot plot:")
print(plot22)

# Additional examples with seaborn
print("\n" + "="*50)
print("SEABORN EXAMPLES")
print("="*50)

# Seaborn plots for comparison
plt.figure(figsize=(10, 6))
sns.boxplot(data=cars_cyl, x='Cylinders', y='cty', hue='Year')
plt.title('Seaborn Box Plot')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=cars_cyl, x='Cylinders', y='cty')
plt.title('Seaborn Violin Plot')
plt.show()

print("\nBasic graphs creation complete!") 