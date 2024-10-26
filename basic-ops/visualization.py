# Sample Univariate Visualization in Python - Single Column

from pickle import FALSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# set default theme
sns.set_theme()

# Import your data into Python
df = pd.read_csv("../dataset/AmesHousingDataset.csv")
print(df.index)

# --------------------------------------- UNIVARIATE ANALYSIS ------------------------------

# 1.1 Box Plot
sns.boxplot(df['Year Built'])   # alternative is plt.boxplot(df['Year Built'])
plt.title('1. Box Plot of Year Built')
plt.show()

#1.2 strip plot is used to visualize the distribution of data points of a single variable
sns.stripplot(y=df['Year Built'])
plt.title('2. Strip Plot of Year Built')
plt.show()

#1.3 Swarm Plot - a visualization technique for univariate data to view the spread of values 
# in a continuous variable.
# For Year Built
sns.swarmplot(x=df['Year Built'])
plt.title('3. Swarm Plot of Year Built')
plt.show()

# For Neighborhood
sns.swarmplot(x=df['Neighborhood'])
plt.title('4. Swarm Plot of Neighborhood')
plt.show()

#1.4 Histograms
plt.hist(df['Year Built'])
plt.title('5. Histogram of Year Built')
plt.show()

# 1.5 SNS distplot to plot a histogram
sns.distplot(df['Year Built'],kde=FALSE, color='blue',bins=5)
plt.title('6. Dist Plot of Year Built with 5 bins')
plt.show()

# 1.6 countplot - visualizing categorical variables
sns.countplot(df['Gender'])
plt.title('7. Count Plot of Gender (Categorical)')
plt.show()

# --------------------------------------- BIVARIATE ANALYSIS -----------------------------

#2.1 Boxplot - visualize the min, max, median, IQR, outliers of a variable
# Covid Severity 1-> Mild; 2-> Moderate; 3->Severe; 4-> Undetermined
sns.boxplot(x=df['Lot Config'],y=df['Neighborhood'],data=df) 
plt.title('8. Box Plot of Covid Severity vs Neighborhood')
plt.show()

#2.2 Scatter Plot
# Visualize the relationship between two variables
sns.scatterplot(x=df['Neighborhood'],y=df['Year Built'])
plt.title('9. Scatter Plot of Year Built vs Neighborhood')
plt.show()

# Hue will indicate which field will have the color coding
sns.scatterplot(x=df['Neighborhood'],y=df['Year Built'],hue=df['Garage Type'])
plt.title('10. Scatter Plot of Year Built vs Neighborhood vs Garage Type (hue value)')
plt.show()

# #2.3 FacetGrid 
# # Gender vs Discharge Type distribution plot
# g = sns.FacetGrid(df, col="Gender", height=6.5, aspect=.85)
# g.map(sns.histplot, "DischargeType")
# plt.title('11. Facet Grid of Gender vs Discharge Type')
# plt.show()

#----------------------------- MULTIVARIATE ANALYSIS ----------------------------------

# # DischargeTypeCategorical vs Year Built vs Gender distribution plot
# g = sns.FacetGrid(df, col="DischargeTypeCategorical", hue="Gender", margin_titles=True, height=6.5, aspect=.85)
# g.map(sns.histplot, "Year Built")
# plt.title('12. Facet Grid of Gender vs Year Built vs Discharge Type')
# plt.show()

# # 2.4 lmplot
# # Year Built vs Gender vs DischargeType
# sns.lmplot(data=df, x="Year Built", y="DischargeType",hue="Gender")
# plt.title('13. lmplot of Year Built vs Discharge Type vs Gender (hue)')
# plt.show()

# # Neighborhood vs Gender vs Year Built
# sns.lmplot(data=df, x="Year Built", y="Neighborhood",hue="Gender")
# plt.title('14. lmplot of Year Built vs Neighborhood vs Gender (hue)')
# #plt.show()

# # Neighborhood vs DischargeTypeCategorical vs Year Built
# sns.lmplot(data=df, x="Neighborhood", y="Year Built",hue="DischargeTypeCategorical")
# plt.title('15. lmplot of Year Built vs Neighborhood vs Discharge Type (hue)')
# #plt.show()



