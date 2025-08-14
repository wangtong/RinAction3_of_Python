"""
------------------------------------------------------------
Python in Action: Chapter 4
Getting started with graphs
requires packages: pandas, plotnine
------------------------------------------------------------
"""
from plotnine import *
import pandas as pd
from plotnine.data import *

CPS85 = pd.read_csv('./datasets/CPS85.csv')
CPS85 = CPS85.drop(['rownames'], axis=1)
CPS85 = CPS85[CPS85['wage'] < 40]

# Figure 4.1 Mapping worker experience and wages to the x and y axes
ggplot(data = CPS85, mapping = aes(x = 'exper', y = 'wage'))

# Figure 4.2 Scatterplot of worker experience vs wages
(
    ggplot(data=CPS85, mapping=aes(x='exper', y='wage')) +
    geom_point()
)

# Figure 4.3 Scatterplot of worker experience vs. wages with outlier removed
CPS85 = CPS85[CPS85['wage'] < 40]
(
    ggplot(data=CPS85, mapping=aes(x='exper', y='wage')) +
    geom_point()
)

# Figure 4.4 Scatterplot of worker experience vs. wages with outlier removed
# with modified point color, transparency, and point size
(
    ggplot(data=CPS85, mapping=aes(x='exper', y='wage')) +
    geom_point(color='cornflowerblue', alpha=0.7, size=1.5) +
    theme_bw()
)

# Figure 4.5 Scatterplot of worker experience vs. wages
# with a line of best fit
(
    ggplot(data=CPS85, mapping=aes(x='exper', y='wage')) +
    geom_point(color='cornflowerblue', alpha=0.7, size=1.5) +
    theme_bw() +
    geom_smooth(method='lm')
)

# Figure 4.6 Scatterplot of worker experience vs. wages
# with points colored by sex and separate line of best
# fit for men and women.
(
    ggplot(data=CPS85, 
    mapping=aes(x='exper', y='wage',
    color='sex', shape='sex', linetype='sex')) +
    geom_point(alpha=0.7, size=1.5) +
    geom_smooth(method='lm', se=False, size=1.5) +
    theme_bw()
)

# Figure 4.7 Scatterplot of worker experience vs. wages
# with custom x- and y-axes and custom color mappings for sex.
(
    ggplot(data=CPS85,
    mapping=aes(x='exper', y='wage',
    color='sex', shape='sex', linetype='sex')) +
    geom_point(alpha=0.7, size=1.5) +
    geom_smooth(method='lm', se=False, size=1.5) +
    scale_x_continuous(breaks=range(0, 61, 10)) +
    scale_y_continuous(breaks=range(0, 31, 5)) +
    scale_color_manual(values=['#CD5555', '#6495ED']) +
    theme_bw()
)

# Figure 4.8 Scatterplot of worker experience vs. wages with
# custom x- and y-axes and custom color mappings for sex.

(
    ggplot(data=CPS85,
    mapping=aes(x='exper', y='wage',
    color='sex', shape='sex', linetype='sex')) +
    geom_point(alpha=0.7, size=1.5) +
    geom_smooth(method='lm', se=False, size=1.5) +
    scale_x_continuous(breaks=range(0, 61, 10)) +
    scale_y_continuous(breaks=range(0, 31, 5),
    labels=lambda x: [f"${i}" for i in x] ) +
    scale_color_manual(values=['#CD5555', '#6495ED']) +
    theme_bw()
)

# Figure 4.9 Scatterplot of worker experience vs. wages with
# custom x- and y-axes and custom color mappings for sex.
# Separate graphs (facets) are provided for each of 8 job sectors.
(
    ggplot(data=CPS85,
    mapping=aes(x='exper', y='wage',
    color='sex', shape='sex', linetype='sex')) +
    geom_point(alpha=0.7) +
    geom_smooth(method='lm', se=False) +
    scale_x_continuous(breaks=range(0, 61, 10)) +
    scale_y_continuous(breaks=range(0, 31, 5),
    labels=lambda x: [f"${i}" for i in x] ) +
    scale_color_manual(values=['#CD5555', '#6495ED']) +
    facet_wrap('~sector') +
    theme_bw()
)

# Figure 4.10 Scatterplot of worker experience vs. wages
# with separate graphs (facets) for each of 8 job sectors
# and custom titles and labels.
(
    ggplot(data=CPS85,
    mapping=aes(x='exper', y='wage',
    color='sex', shape='sex', linetype='sex')) +
    geom_point(alpha=0.7) +
    geom_smooth(method='lm', se=False) +
    scale_x_continuous(breaks=range(0, 61, 10)) +
    scale_y_continuous(breaks=range(0, 31, 5),
    labels=lambda x: [f"${i}" for i in x] ) +
    scale_color_manual(values=['#CD5555', '#6495ED']) +
    facet_wrap('~sector') +
    labs(title='Relationship between wages and experience',
    subtitle='Current Population Survey',
    caption='source: http://mosaic-web.org/',
    x='Years of Experience',
    y='Hourly Wage') +
    theme_bw()
 )
# Figure 4.11 Scatterplot of worker experience vs. wages
# with separate graphs (facets) for each of 8 job sectors
# and custom titles and labels, and a cleaner theme.
(
ggplot(data = CPS85,
       mapping = aes(x = 'exper', y = 'wage', color = 'sex', shape='sex',
                     linetype = 'sex')) +
  geom_point(alpha = .7) +
  geom_smooth(method = "lm", se = False) +
  scale_x_continuous(breaks = range(0, 60, 10)) +
  scale_y_continuous(breaks = range(0, 30, 5),
                     labels = lambda x: [f"${i}" for i in x] ) +
  scale_color_manual(values = ['#CD5555', '#6495ED']) +
  facet_wrap('~sector') +
  labs(title = "Relationship between wages and experience",
       subtitle = "Current Population Survey",
       caption = "source: http://mosaic-web.org/",
       x = " Years of Experience",
       y = "Hourly Wage",
       color = "Gender", shape="Gender", linetype="Gender") +
  theme_minimal()
)

# -- Section 4.2 ggplot2 details

# Figure 4.12. Scatterplot of experience and wage by sex,
# where aes(color=sex) is placed in the ggplot() function.
# The mapping is applied to both the geom_point() and geom_smooth(),
# producing separate point colors for males and females,
# along with separate lines of best fit
(
    ggplot(CPS85,
    mapping = aes(x = 'exper', y = 'wage', color = 'sex')) +
    geom_point(alpha = .7, size = 1.5) +
    geom_smooth(method = "lm", se = False, size = 1)
)
# Figure 4.13 Scatterplot of experience and wage by sex,
# where aes(color=sex) is placed in the geom_point() function.
# The mapping is applied to point color producing separate point
# colors for men and women, but a single line of best fit
# for or all workers.
(
    ggplot(CPS85, aes(x = 'exper', y = 'wage')) +
    geom_point(aes(color = 'sex'), alpha = .7, size = 1.5) +
    geom_smooth(method = "lm", se = False, size = 1)
)
# Listing 4.1 Using a ggplot2 graph as an object

CPS85 = pd.read_csv('./datasets/CPS85.csv')
CPS85 = CPS85.drop(['rownames'], axis=1)
CPS85 = CPS85[CPS85['wage'] < 40]