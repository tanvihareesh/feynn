#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("Indian automoble buying behavour study 1.0.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


for column in ['Profession','Marrital Status','Education','Personal loan','House Loan','Wife Working','Make']:
  print(column,':',df[column].unique())


# Visualization

# In[9]:


# Visualization based on profession
sns.countplot(data=df, x='Profession',palette="Paired")
plt.title('Profession Analysis')
plt.xlabel('Profession')
plt.ylabel('Count')
plt.show()


# Salaried people are getting more attracted to the EV vehicles 

# In[10]:


# Histogram based on customer age
sns.histplot(data=df, x='Age', bins=10,color='blue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# People of age between 32 and 38 are getting more interests in purchasing EV vehicle

# In[11]:


# Visualization based on marital status
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Marrital Status',palette="tab10")
plt.title('Marital Status Analysis')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()


# As mentioned earlier, we people having age between 32 and 38 are more interested in EV vehicles, so they are generally married at that age

# In[13]:


# Visualization based on educational qualification
plt.figure(figsize=(5, 3))
sns.countplot(data=df, x='Education',palette="cubehelix")
plt.title('Education Analysis')
plt.xlabel('Education')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# So, we can say people with more educational qualification are more aware of the importance of EV vehicles in our society

# In[14]:


# Total Salary Distribution visualization
plt.figure(figsize=(5, 3))
sns.histplot(data=df, x='Total Salary', bins=10,color='green', edgecolor='black')
plt.title('Total Salary Distribution')
plt.xlabel('Total Salary')
plt.ylabel('Count')
plt.show()


# People, earning between 800000/- and 3200000/- are main buyers 

# In[15]:


#Vehicle Make Analysis
plt.figure(figsize=(5, 3))
sns.countplot(data=df, x='Make',palette="tab10")
plt.title('Vehicle Make Analysis')
plt.xlabel('Make')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Baleno and SUV are main market occupiers

# In[19]:


# Scatterplot to visualize the relation between age and price of car owned 
plt.xlabel('Age')
plt.ylabel('Price of car owned')
plt.scatter(df['Age'],df['Price'])


# In[20]:


# Scatterplot to visualize the relation between total salary and price of car owned 
plt.xlabel('total salary')
plt.ylabel('Price of car owned')
plt.scatter(df['Total Salary'],df['Price'])


# In[21]:


# Scatterplot to visualize the relation between marital status and price of car owned 
plt.xlabel('Marrital Status')
plt.ylabel('Price of car owned')
plt.scatter(df['Marrital Status'],df['Price'])


# In[22]:


# Scatterplot to visualize the relation between profession and price of car owned 
plt.xlabel('Profession')
plt.ylabel('Price of car owned')
plt.scatter(df['Profession'],df['Price'])


# In[23]:


# Scatterplot to visualize the relation between education and price of car owned 
plt.xlabel('Education')
plt.ylabel('Price of car owned')
plt.scatter(df['Education'],df['Price'])


# In[24]:


# Scatterplot to visualize the relation between personal loan and price of car owned 
plt.xlabel('Personal loan')
plt.ylabel('Price of car owned')
plt.scatter(df['Personal loan'],df['Price'])


# In[25]:


# Heatmap of Correlation
sns.heatmap(df.corr(), annot=True)


# In[23]:


# Pair Plot
sns.pairplot(df)

