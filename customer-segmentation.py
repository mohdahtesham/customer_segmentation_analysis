import pandas as pd
import numpy as np
import seaborn as sb
import os
import matplotlib.pyplot as plt

#loading data in a dataframe using pandas

df = pd.read_csv("C:/Users/megha/OneDrive/Documents/Customer Segmentation/Mall_Customers.csv")

#to see if loaded correctly
print(df)

#to print a concise summary of the dataframe
df.info()
#to get info about the 1st 5 rows
df.head()
#Describe - atleast there is one numeric column, it will show
#us descriptive statistics of all num columns
df.describe() 
#no of rows and columns in a dataframe -- no paranthesis
df.shape
#to get a gist about columns, dont use paranthesis
df.columns
#datatypes -- no columns
df.dtypes

#Dataframe has both- METHODS and ATTRIBUTES
#METHODS are the ones with paranthesis
#ATTRIBUTES don't have ()

help(pd.DataFrame.drop)
#dropping the customer id column to avoid confusion, also it seems useless
#using the inplace = true will ensure the changes are saved to the same variable

df.drop(["CustomerID"], axis = 1, inplace = True)

df.head()

#violin plots are the most visual and descriptive way to visualize a data distribution
#show a lot of info about the data, nice to look at unlike the regular bar charts
#show several descriptive stats, including median and IQ range
#also denote outliers
#thick line indicates the IQ range

#wider sections rep higher probability that members of population will take on the data


plt.figure(figsize=(10,6))
plt.title("Ages Frequency")
sb.axes_style("darkgrid")
sb.violinplot(y=df["Age"])
plt.show()


#viz spending scores and annual income using boxplots
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
sb.boxplot(y=df["Spending Score (1-100)"], color="red")
plt.subplot(1,2,2)
sb.boxplot(y=df["Annual Income (k$)"])
plt.show()

#comaparing what genders are dominant
genders = df.Gender.value_counts()
sb.set_style("darkgrid")
plt.figure(figsize=(10,4))
sb.barplot(x=genders.index, y=genders.values)
plt.show()


age_18_25 = df.Age[(df.Age <= 25) & (df.Age >= 18)]
age_26_35 = df.Age[(df.Age <= 35) & (df.Age >= 26)]
age_36_45 = df.Age[(df.Age <= 45) & (df.Age >= 36)]
age_46_55 = df.Age[(df.Age <= 55) & (df.Age >= 46)]
age_55above = df.Age[df.Age >= 56]

x = ["18-25","26-35","36-45","46-55","55+"]
y = [len(age_18_25.values),len(age_26_35.values),len(age_36_45.values),len(age_46_55.values),len(age_55above.values)]

plt.figure(figsize=(15,6))
sb.barplot(x=x, y=y, palette="rocket")
plt.title("Number of Customer and Ages")
plt.xlabel("Age")
plt.ylabel("Number of Customer")
plt.show()

#visualizing spending scores in groups

ss1_20 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 1) & (df["Spending Score (1-100)"] <= 20)]
ss21_40 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 21) & (df["Spending Score (1-100)"] <= 40)]
ss41_60 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 41) & (df["Spending Score (1-100)"] <= 60)]
ss61_80 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 61) & (df["Spending Score (1-100)"] <= 80)]
ss81_100 = df["Spending Score (1-100)"][(df["Spending Score (1-100)"] >= 81) & (df["Spending Score (1-100)"] <= 100)]

ssx = ["1-20", "21-40", "41-60", "61-80", "81-100"]
ssy = [len(ss1_20.values), len(ss21_40.values), len(ss41_60.values), len(ss61_80.values), len(ss81_100.values)]

plt.figure(figsize=(15,6))
sb.barplot(x=ssx, y=ssy, palette="nipy_spectral_r")
plt.title("Spending Scores")
plt.xlabel("Score")
plt.ylabel("Number of Customer Having the Score")
plt.show()

#grouping annual incomes
ai0_30 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 0) & (df["Annual Income (k$)"] <= 30)]
ai31_60 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 31) & (df["Annual Income (k$)"] <= 60)]
ai61_90 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 61) & (df["Annual Income (k$)"] <= 90)]
ai91_120 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 91) & (df["Annual Income (k$)"] <= 120)]
ai121_150 = df["Annual Income (k$)"][(df["Annual Income (k$)"] >= 121) & (df["Annual Income (k$)"] <= 150)]

aix = ["$ 0 - 30,000", "$ 30,001 - 60,000", "$ 60,001 - 90,000", "$ 90,001 - 120,000", "$ 120,001 - 150,000"]
aiy = [len(ai0_30.values), len(ai31_60.values), len(ai61_90.values), len(ai91_120.values), len(ai121_150.values)]

plt.figure(figsize=(15,6))
sb.barplot(x=aix, y=aiy, palette="Set2")
plt.title("Annual Incomes")
plt.xlabel("Income")
plt.ylabel("Number of Customer")
plt.show()

#Kmeans

### Kmeans algorithm is an iterative algorithm that tries to partition the dataset into Kpre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group.
###  It tries to make the inter-cluster data points as similar as possible while also keeping the clusters as different (far) as possible. 

###Goal:
# Get a meaningful intuition of the structure of the data we’re dealing with.



#### plotted Within Cluster Sum Of Squares (WCSS) against the the number of clusters (K Value) to figure out the optimal number of clusters value. WCSS measures sum of distances of observations from their cluster centroids
# main goal = maximize the number of clusters

from sklearn.cluster import KMeans
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(df.iloc[:,1:])
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))    
plt.grid()
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()

#optimal value of K can be 4 or 5
#I chose 5 here
#creating the 3d plot using K means clustering

km = KMeans(n_clusters=5)
clusters = km.fit_predict(df.iloc[:,1:])
df["label"] = clusters

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.Age[df.label == 0], df["Annual Income (k$)"][df.label == 0], df["Spending Score (1-100)"][df.label == 0], c='blue', s=60)
ax.scatter(df.Age[df.label == 1], df["Annual Income (k$)"][df.label == 1], df["Spending Score (1-100)"][df.label == 1], c='red', s=60)
ax.scatter(df.Age[df.label == 2], df["Annual Income (k$)"][df.label == 2], df["Spending Score (1-100)"][df.label == 2], c='green', s=60)
ax.scatter(df.Age[df.label == 3], df["Annual Income (k$)"][df.label == 3], df["Spending Score (1-100)"][df.label == 3], c='orange', s=60)
ax.scatter(df.Age[df.label == 4], df["Annual Income (k$)"][df.label == 4], df["Spending Score (1-100)"][df.label == 4], c='purple', s=60)
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()


