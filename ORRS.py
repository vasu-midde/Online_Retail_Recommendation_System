#!/usr/bin/env python
# coding: utf-8

# In[78]:

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics.pairwise import cosine_similarity

# In[3]:

# Load the dataset from an Excel file
data = pd.read_excel("C:/Users/chintu/Downloads/data.xlsx")

# In[4]:

# Display information about the dataset
data.info()

# In[5]:

# Print the shape of the dataset
data.shape

# In[6]:

# Get a statistical summary of the dataset
data.describe()

# In[7]:

# Show the first few rows of the dataset
data.head()

# In[8]:

# Remove leading and trailing spaces from the 'Description' column
data["Description"] = data["Description"].str.strip()

# In[9]:

# Drop rows where 'InvoiceNo' is NaN (removing duplicates)
data.dropna(axis=0, subset=["InvoiceNo"], inplace=True)

# In[10]:

# Drop rows where 'CustomerID' is NaN
data.dropna(axis=0, subset=["CustomerID"], inplace=True)

# In[11]:

# Convert 'InvoiceNo' to string type
data["InvoiceNo"] = data["InvoiceNo"].astype('str')

# In[12]:

# Filter out credit transactions by removing rows with 'InvoiceNo' containing 'C'
data = data[~data['InvoiceNo'].str.contains('C')]

# In[13]:

# Display the first few rows of the filtered dataset
data.head()

# In[14]:

# Print the shape of the filtered dataset
data.shape

# In[15]:

# Count the occurrences of each 'StockCode'
data["StockCode"].value_counts()

# In[16]:

# Display data types of each column
data.dtypes

# In[17]:

# Calculate the total sales by multiplying 'Quantity' and 'UnitPrice'
data["Sales"] = data["Quantity"] * data["UnitPrice"]

# In[18]:

# Show the updated dataset with sales information
data

# In[19]:

# Extract the month from 'InvoiceDate'
data["Month"] = data["InvoiceDate"].dt.month

# In[20]:

# Display the first few rows of the dataset with the 'Month' column
data.head()

# In[21]:

# Show the last few rows of the dataset
data.tail()

# Most Popular Products Globally

# In[109]:

# Aggregate total quantity sold for each product globally
global_popularity = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
global_popularity

# In[110]:

# Plot the top 10 globally popular products
plt.figure(figsize=(12,6))
global_popularity.head(10).plot(kind='bar')
plt.title('Top 10 Products by Global Sales')
plt.xlabel('Product Description')
plt.ylabel('Total Quantity Sold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
print("Top 10 Products by Global Sales:")
print(global_popularity.head(10))

# Popular Products by Country

# In[114]:

# Aggregate total quantity sold by country and product
country_popularity = data.groupby(['Country', 'Description'])['Quantity'].sum().reset_index()
country_popularity = country_popularity.sort_values(['Country', 'Quantity'], ascending=[True, False])
country_popularity = country_popularity.groupby('Country').first().reset_index()

# In[115]:

# Display the most popular product for each country
country_popularity

# In[117]:

# Plot the most popular product in the top 10 countries
plt.figure(figsize=(12,6))
sns.barplot(x='Country', y='Quantity', data=country_popularity.head(10))
plt.title("Top Product by Quantity in Leading Countries")
plt.xlabel('Country')
plt.ylabel('Quantity of Top Product')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
print("\nTop Product by Quantity in Each Country:")
print(country_popularity[['Country', 'Description', 'Quantity']].head(10))

# Monthly Popular Products

# In[118]:

# Aggregate total quantity sold by month and product
monthly_popularity = data.groupby(['Month', 'Description'])['Quantity'].sum().reset_index()
monthly_popularity = monthly_popularity.sort_values(['Month', 'Quantity'], ascending=[True, False])
monthly_popularity = monthly_popularity.groupby('Month').first().reset_index()
monthly_popularity

# In[119]:

# Plot the quantity of the most popular product each month
plt.figure(figsize=(12,6))
sns.lineplot(x='Month', y='Quantity', data=monthly_popularity)
plt.title("Monthly Quantity of Leading Products")
plt.xlabel("Month")
plt.ylabel("Quantity of Top Product")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
print("\nTop Product Each Month:")
print(monthly_popularity[['Month', 'Description', 'Quantity']])

# Additional Analysis

# In[120]:

# Sort quantities in descending order
data["Quantity"].sort_values(ascending=False)

# In[121]:

# Count transactions by country
data['Country'].value_counts()

# In[122]:

# Create a pivot table of sales by customer and product
pivot_table = data.pivot_table(index='CustomerID', columns='StockCode', values='Sales', aggfunc="sum", fill_value=0)
pivot_table.head()

# In[123]:

# Compute the correlation matrix of the pivot table
correlation_matrix = pivot_table.corr()

# In[124]:

# Plot a heatmap of the correlation matrix for the first 20 products
plt.figure(figsize=(12,10))
sns.heatmap(correlation_matrix.iloc[:20, :20], annot=False, cmap="coolwarm")
plt.title("Product Correlation Heatmap for First 20 Products")
plt.show()

# Recommendation Functions

# In[125]:

def recommend_products(customer_id, product_code, pivot_table, correlation_matrix, top_n=5):
    if product_code not in correlation_matrix.columns:
        return pd.Series()
    similar_products = correlation_matrix[product_code].sort_values(ascending=False)
    similar_products = similar_products.iloc[1:top_n+1]
    
    customer_sales = pivot_table.loc[customer_id]
    
    predicted_scores = []
    for product in similar_products.index:
        if product in customer_sales.index:
            score = similar_products[product] * customer_sales[product]
            predicted_scores.append(score)
        else:
            predicted_scores.append(0)
    recommendations = pd.Series(predicted_scores, index=similar_products.index)
    return recommendations.sort_values(ascending=False)

# In[126]:

# Example recommendation
customer_id = 17850.0
product_code = '84029G'
recommendations = recommend_products(customer_id, product_code, pivot_table, correlation_matrix)
print(f"Recommendations for customer {customer_id} based on product {product_code}:")
print(recommendations)

# In[127]:

# Display the first few rows of the dataset
data.head()

# In[128]:

def get_top_recommendations_for_customer(customer_id, pivot_table, correlation_matrix, top_n=5):
    customer_sales = pivot_table.loc[customer_id]
    purchased_products = customer_sales[customer_sales > 0].index
    
    all_recommendations = pd.Series()
    for product in purchased_products:
        product_recommendations = recommend_products(customer_id, product, pivot_table, correlation_matrix)
        all_recommendations = all_recommendations.add(product_recommendations, fill_value=0)
    
    all_recommendations = all_recommendations[~all_recommendations.index.isin(purchased_products)]
    
    return all_recommendations.sort_values(ascending=False).head(top_n)

# Example usage
customer_id = 13047.0
top_recommendations = get_top_recommendations_for_customer(customer_id, pivot_table, correlation_matrix)
print(f"\nTop recommendations for customer {customer_id}:")
print(top_recommendations)

# Note: The recommendation system is based on item-based collaborative filtering. It may not be highly accurate due to limited data and lack of detailed customer information.
