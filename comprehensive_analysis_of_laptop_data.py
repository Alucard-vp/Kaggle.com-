#!/usr/bin/env python
# coding: utf-8

# In[16]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 
# 
# This notebook provides an in-depth analysis of a dataset containing various features of laptops and their corresponding prices. By exploring the relationships between different attributes such as company, type, RAM, CPU, GPU, and storage, we aim to uncover insights that can help users understand the factors affecting laptop prices. Below is the table of contents outlining the key analyses performed:
# 
# ## Table of Contents
# 
# 1. Average Price by Company
# 2. Distribution of Operating Systems
# 3. Average Price by RAM
# 4. Correlation Heatmap (only numeric columns)
# 5. Distribution of Laptop Types by Company
# 6. Price Distribution Across RAM Sizes
# 7. Average Price by CPU Brand
# 8. Average Price by GPU Brand
# 9. Price vs Weight
# 10. Operating System Distribution by Company
# 11. Price by Touchscreen and IPS Features
# 

# In[17]:


df=pd.read_csv('/kaggle/input/laptop-prices/laptop-price.csv')
df.head()


# In[18]:


print(df.shape)  
print(df.columns.tolist())  
print(df.isnull().sum())  
print(df.describe()) 


# # 1. Average Price by Company
# 

# In[19]:


import pandas as pd
import plotly.express as px


# Average Price by Company
avg_price_by_company = df.groupby('Company')['Price'].mean().sort_values(ascending=False).reset_index()
fig1 = px.bar(avg_price_by_company, x='Company', y='Price', title='Average Laptop Price by Company',
              labels={'Price': 'Average Price', 'Company': 'Laptop Company'})
fig1.show(renderer='iframe_connected')


# # 2. Distribution of Operating Systems
# 

# In[20]:


# Distribution of Operating Systems
os_distribution = df['os'].value_counts().reset_index()
os_distribution.columns = ['Operating System', 'Count']
fig2 = px.pie(os_distribution, names='Operating System', values='Count',
              title='Operating System Distribution', hole=0.3)
fig2.show(renderer='iframe_connected')



# # 3. Average Price by RAM
# 

# In[21]:


# Average Price by RAM
avg_price_by_ram = df.groupby('Ram')['Price'].mean().sort_values(ascending=False).reset_index()
fig3 = px.bar(avg_price_by_ram, x='Ram', y='Price', title='Average Laptop Price by RAM Size',
              labels={'Price': 'Average Price', 'Ram': 'RAM (GB)'})
fig3.show(renderer='iframe_connected')



# # 4. Correlation Heatmap (only numeric columns)
# 

# In[22]:


# Correlation Heatmap (only numeric columns)
numeric_columns = df.select_dtypes(include='number')
correlation = numeric_columns.corr()
fig4 = px.imshow(correlation, title='Correlation Matrix', color_continuous_scale='Viridis',
                 labels=dict(color="Correlation"))
fig4.show(renderer='iframe_connected')


# # 5. Distribution of Laptop Types by Company
# 

# In[23]:


# 1. Distribution of Laptop Types by Company
laptop_type_distribution = df.groupby(['Company', 'TypeName']).size().reset_index(name='Count')
fig1 = px.bar(laptop_type_distribution, x='Company', y='Count', color='TypeName',
              title="Distribution of Laptop Types by Company",
              labels={'Count': 'Number of Laptops', 'Company': 'Laptop Company'})
fig1.show(renderer='iframe_connected')


# # 6. Price Distribution Across RAM Sizes
# 

# In[24]:


# 2. Price Distribution Across RAM Sizes
fig2 = px.box(df, x='Ram', y='Price', title="Price Distribution Across RAM Sizes",
              labels={'Price': 'Laptop Price', 'Ram': 'RAM (GB)'})
fig2.show(renderer='iframe_connected')



# # 6. Average Price by CPU Brand
# 

# In[25]:


# 3. Average Price by CPU Brand
cpu_price_analysis = df.groupby('Cpu brand')['Price'].mean().reset_index()
fig3 = px.bar(cpu_price_analysis, x='Cpu brand', y='Price',
              title="Average Price by CPU Brand",
              labels={'Price': 'Average Price', 'Cpu brand': 'CPU Brand'})
fig3.show(renderer='iframe_connected')


# # 8. Average Price by GPU Brand
# 

# In[26]:


# 4. Average Price by GPU Brand
gpu_price_analysis = df.groupby('Gpu brand')['Price'].mean().reset_index()
fig4 = px.bar(gpu_price_analysis, x='Gpu brand', y='Price',
              title="Average Price by GPU Brand",
              labels={'Price': 'Average Price', 'Gpu brand': 'GPU Brand'})
fig4.show(renderer='iframe_connected')


# # 9. Price vs Weight
# 

# In[27]:


# 5. Price vs Weight
fig5 = px.scatter(df, x='Weight', y='Price', color='Company',
                  title="Relationship Between Weight and Price",
                  labels={'Price': 'Laptop Price', 'Weight': 'Weight (kg)'})
fig5.show(renderer='iframe_connected')


# # 10. Operating System Distribution by Company
# 

# In[28]:


# 6. Operating System Distribution by Company
os_distribution_company = df.groupby(['Company', 'os']).size().reset_index(name='Count')
fig6 = px.bar(os_distribution_company, x='Company', y='Count', color='os',
              title="Operating System Distribution by Company",
              labels={'Count': 'Number of Laptops', 'Company': 'Laptop Company'})
fig6.show(renderer='iframe_connected')


# # 11. Price by Touchscreen and IPS Features
# 

# In[29]:


# 7. Price by Touchscreen and IPS Features
touchscreen_ips_analysis = df.groupby(['Touchscreen', 'Ips'])['Price'].mean().reset_index()
fig7 = px.bar(touchscreen_ips_analysis, x='Touchscreen', y='Price', color='Ips',
              title="Price Analysis by Touchscreen and IPS Features",
              labels={'Price': 'Average Price', 'Touchscreen': 'Touchscreen Support', 'Ips': 'IPS Support'})
fig7.show(renderer='iframe_connected')

