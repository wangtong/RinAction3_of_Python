#------------------------------------------------------------#
# R in Action (3rd ed): Chapter 4 - Python Translation      #
# Getting started with graphs                                #
# requires packages: plotnine, pandas, numpy                #
# pip install plotnine pandas numpy matplotlib seaborn      #
#------------------------------------------------------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotnine import *
import warnings
warnings.filterwarnings('ignore')

## Chapter 04
## Getting started with graphs

# Create CPS85 dataset (similar to mosaicData::CPS85)
np.random.seed(42)
n = 534

# Generate synthetic CPS85-like data
cps85_data = {
    'wage': np.random.gamma(2, 5, n),  # Wage data
    'exper': np.random.poisson(15, n),  # Experience
    'sector': np.random.choice(['manag', 'manuf', 'HiTec', 'const', 'sales', 'clerc', 'serv', 'other'], n),
    'sex': np.random.choice(['M', 'F'], n, p=[0.6, 0.4])
}

CPS85 = pd.DataFrame(cps85_data)
CPS85['wage'] = np.clip(CPS85['wage'], 0, 40)  # Cap wages at 40 like in R code

print("CPS85 dataset created:")
print(CPS85.head())
print(f"Dataset shape: {CPS85.shape}")

# -- Section 4.1 Creating a graph with plotnine

# Figure 4.1 Mapping worker experience and wages to the x and y axes
print("\n" + "="*50)
print("BASIC PLOTNINE EXAMPLES")
print("="*50)

plot1 = ggplot(data=CPS85, mapping=aes(x='exper', y='wage'))
print("Basic mapping (no geom):")
print(plot1)

# Figure 4.2 Scatterplot of worker experience vs wages
plot2 = (ggplot(data=CPS85, mapping=aes(x='exper', y='wage')) +
         geom_point())
print("\nScatterplot:")
print(plot2)

# Remove outliers (wages > 40)
CPS85 = CPS85[CPS85['wage'] < 40]

# Figure 4.3 Scatterplot with outlier removed
plot3 = (ggplot(data=CPS85, mapping=aes(x='exper', y='wage')) +
         geom_point())
print("\nScatterplot with outliers removed:")
print(plot3)

# Figure 4.4 Modified point color, transparency, and size
plot4 = (ggplot(data=CPS85, mapping=aes(x='exper', y='wage')) +
         geom_point(color="cornflowerblue", alpha=0.7, size=1.5) +
         theme_bw())
print("\nStyled scatterplot:")
print(plot4)

# Figure 4.5 Scatterplot with line of best fit
plot5 = (ggplot(data=CPS85, mapping=aes(x='exper', y='wage')) +
         geom_point(color="cornflowerblue", alpha=0.7, size=1.5) +
         theme_bw() +
         geom_smooth(method="lm"))
print("\nScatterplot with regression line:")
print(plot5)

# Figure 4.6 Points colored by sex with separate lines
plot6 = (ggplot(data=CPS85, 
                mapping=aes(x='exper', y='wage', 
                           color='sex', shape='sex', linetype='sex')) +
         geom_point(alpha=0.7, size=1.5) +
         geom_smooth(method="lm", se=False, size=1.5) +
         theme_bw())
print("\nScatterplot by sex:")
print(plot6)

# Figure 4.7 Custom axes and color mappings
plot7 = (ggplot(data=CPS85,
                mapping=aes(x='exper', y='wage',
                           color='sex', shape='sex', linetype='sex')) +
         geom_point(alpha=0.7, size=1.5) +
         geom_smooth(method="lm", se=False, size=1.5) +
         scale_x_continuous(breaks=range(0, 61, 10)) +
         scale_y_continuous(breaks=range(0, 31, 5)) +
         scale_color_manual(values=["indianred3", "cornflowerblue"]) +
         theme_bw())
print("\nCustom styling:")
print(plot7)

# Figure 4.8 Dollar format for wages
try:
    from mizani.formatters import dollar_format
    plot8 = (ggplot(data=CPS85,
                    mapping=aes(x='exper', y='wage',
                               color='sex', shape='sex', linetype='sex')) +
             geom_point(alpha=0.7, size=1.5) +
             geom_smooth(method="lm", se=False, size=1.5) +
             scale_x_continuous(breaks=range(0, 61, 10)) +
             scale_y_continuous(breaks=range(0, 31, 5),
                               labels=dollar_format()) +
             scale_color_manual(values=["indianred3", "cornflowerblue"]) +
             theme_bw())
    print("\nWith dollar formatting:")
    print(plot8)
except ImportError:
    print("\nMizani not available for dollar formatting")

# Figure 4.9 Faceted by sector
plot9 = (ggplot(data=CPS85,
                mapping=aes(x='exper', y='wage',
                           color='sex', shape='sex', linetype='sex')) +
         geom_point(alpha=0.7) +
         geom_smooth(method="lm", se=False) +
         scale_x_continuous(breaks=range(0, 61, 10)) +
         scale_y_continuous(breaks=range(0, 31, 5)) +
         scale_color_manual(values=["indianred3", "cornflowerblue"]) +
         facet_wrap('~sector') +
         theme_bw())
print("\nFaceted by sector:")
print(plot9)

# Figure 4.10 With custom titles and labels
plot10 = (ggplot(data=CPS85,
                 mapping=aes(x='exper', y='wage',
                            color='sex', shape='sex', linetype='sex')) +
          geom_point(alpha=0.7) +
          geom_smooth(method="lm", se=False) +
          scale_x_continuous(breaks=range(0, 61, 10)) +
          scale_y_continuous(breaks=range(0, 31, 5)) +
          scale_color_manual(values=["indianred3", "cornflowerblue"]) +
          facet_wrap('~sector') +
          labs(title="Relationship between wages and experience",
               subtitle="Current Population Survey",
               caption="source: Synthetic data based on CPS85",
               x="Years of Experience",
               y="Hourly Wage",
               color="Gender", shape="Gender", linetype="Gender") +
          theme_bw())
print("\nWith custom labels:")
print(plot10)

# Figure 4.11 With minimal theme
plot11 = (ggplot(data=CPS85,
                 mapping=aes(x='exper', y='wage', color='sex', 
                            shape='sex', linetype='sex')) +
          geom_point(alpha=0.7) +
          geom_smooth(method="lm", se=False) +
          scale_x_continuous(breaks=range(0, 61, 10)) +
          scale_y_continuous(breaks=range(0, 31, 5)) +
          scale_color_manual(values=["indianred3", "cornflowerblue"]) +
          facet_wrap('~sector') +
          labs(title="Relationship between wages and experience",
               subtitle="Current Population Survey",
               caption="source: Synthetic data based on CPS85",
               x="Years of Experience",
               y="Hourly Wage",
               color="Gender", shape="Gender", linetype="Gender") +
          theme_minimal())
print("\nWith minimal theme:")
print(plot11)

# -- Section 4.2 plotnine details

print("\n" + "="*50)
print("PLOTNINE DETAILS")
print("="*50)

# Figure 4.12 Color mapping in ggplot() function
plot12 = (ggplot(CPS85, mapping=aes(x='exper', y='wage', color='sex')) +
          geom_point(alpha=0.7, size=1.5) +
          geom_smooth(method="lm", se=False, size=1))
print("\nColor mapping in main ggplot:")
print(plot12)

# Figure 4.13 Color mapping in geom_point() function
plot13 = (ggplot(CPS85, aes(x='exper', y='wage')) +
          geom_point(aes(color='sex'), alpha=0.7, size=1.5) +
          geom_smooth(method="lm", se=False, size=1))
print("\nColor mapping in geom_point only:")
print(plot13)

# Listing 4.1 Using a plotnine graph as an object
myplot = (ggplot(data=CPS85, mapping=aes(x='exper', y='wage')) +
          geom_point())

print("\nBase plot object:")
print(myplot)

myplot2 = myplot + geom_point(size=3, color="blue")
print("\nModified plot object:")
print(myplot2)

final_plot = (myplot + 
              geom_smooth(method="lm") +
              labs(title="Mildly interesting graph"))
print("\nFinal modified plot:")
print(final_plot)

# Common mistake example
mistake_plot = (ggplot(CPS85, aes(x='exper', y='wage', color="blue")) +
                geom_point())
print("\nCommon mistake (color as aesthetic):")
print(mistake_plot)

# Additional examples
print("\n" + "="*50)
print("ADDITIONAL EXAMPLES")
print("="*50)

# Distribution plots
dist_plot = (ggplot(CPS85, aes(x='wage')) +
             geom_histogram(bins=30, fill='skyblue', alpha=0.7) +
             labs(title="Wage Distribution",
                  x="Hourly Wage",
                  y="Count") +
             theme_minimal())
print("\nWage distribution:")
print(dist_plot)

# Box plot by sector
box_plot = (ggplot(CPS85, aes(x='sector', y='wage')) +
            geom_boxplot(fill='lightgreen', alpha=0.7) +
            labs(title="Wage by Sector",
                 x="Sector",
                 y="Hourly Wage") +
            theme_minimal() +
            theme(axis_text_x=element_text(rotation=45, hjust=1)))
print("\nBox plot by sector:")
print(box_plot)

# Density plot by sex
density_plot = (ggplot(CPS85, aes(x='wage', fill='sex', color='sex')) +
                geom_density(alpha=0.5) +
                labs(title="Wage Density by Sex",
                     x="Hourly Wage",
                     y="Density") +
                theme_minimal())
print("\nDensity plot by sex:")
print(density_plot)

print("\nGraphs creation complete!") 