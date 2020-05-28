import numpy as np 
import pandas as pd 
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
o_retail = pd.read_csv("/kaggle/input/online-retail/online_retail.csv")

o_retail['Description'] = o_retail['Description'].str.strip()
o_retail.dropna(axis = 0, subset = ['InvoiceNo'], inplace = True)
o_retail['InvoiceNo'] = o_retail['InvoiceNo'].astype('str')
o_retail = o_retail[~o_retail['InvoiceNo'].str.contains('C')]
#o_retail.head()

basket = (o_retail[o_retail['Country'] =='France'].groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)) 
#basket

basket_sets = basket.applymap(encode_units)
basket_sets.drop('POSTAGE', inplace = True, axis = 1)
basket_sets

frequent_itemsets = apriori( basket_sets, min_support = 0.07, use_colnames = True)
rules = association_rules(frequent_itemsets, metric = 'lift',min_threshold = 1)
rules.head()