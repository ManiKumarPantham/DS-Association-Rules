####################################################################################################
Q1. Kitabi Duniya, a famous book store in India, which was established before Independence, the growth 
of the company was incremental year by year, but due to online selling of books and wide spread 
Internet access its annual growth started to collapse, seeing sharp downfalls, you as 
a Data Scientist help this heritage book store gain its popularity back and increase footfall of 
customers and provide ways the business can improve exponentially, apply Association RuleAlgorithm, 
explain the rules, and visualize the graphs for clear understanding of solution.

####################################################################################################

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder

# Reading the data into python
data = pd.read_csv('D:/Hands on/12_Association Rules/Assignment/book.csv')

# Counting the number of words
count = data.loc[:, :].sum()

# Top 10 items
pop_item = count.sort_values(0, ascending = False).head(10)

# Converting it into Datafreame
pop_item =  pop_item.to_frame()

# Resetting index
pop_item = pop_item.reset_index()

# Columns
pop_item.columns

# Rename
pop_item.rename({'index' : 'book', 0 : 'count'}, axis = 1, inplace = True)

#%matplolib inline
ax = pop_item.plot.barh(x = 'book', y = 'count')
plt.gca().invert_yaxis()
plt.title('Most Popular items')
plt.show()

# Frequent items
frequent_items = apriori(data, min_support = 0.0075, max_len = 4, use_colnames = True)
frequent_items

# Sorting
frequent_items.sort_values('support', ascending = False, inplace = True)

# Association rules
rules = association_rules(frequent_items, metric = 'lift', min_threshold = 1)
rules

# Sorting
rules.sort_values('lift', ascending = False).head(10)

# User defined function
def to_list(i):
    return sorted(list(i))


# Apllying sorting on antecedents and consequents
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
ma_X

# Sorting
ma_X = ma_X.apply(sorted)

# Convering it in to list
rules_set = list(ma_X)

# Unique items
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_set)]

# Empty list
index_rules = []

# User defined function
for i in unique_rules_sets:
    index_rules.append(rules_set.index(i))
    
index_rules

# Redundancy 
rules_no_redundancy = rules.iloc[index_rules, :]
rules_no_redundancy

# Top 10 rules
rules10 = rules_no_redundancy.sort_values('lift', ascending = False).head(10)
rules10

rules10.plot(x = "support", y = "confidence", c = rules10.lift, kind="scatter", s = 12, cmap = plt.cm.coolwarm)


#####################################################################################################
2. A film distribution company wants to target audience based on their likes and dislikes, 
you as a Chief Data Scientist Analyze the data and come up with different rules of movie list 
so that the business objective is achieved.

#####################################################################################################

# Importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Reading the data into python
data = pd.read_csv("D:/Hands on/12_Association Rules/Assignment/my_movies.csv")

# Droping the columns
data.drop(['V1', 'V2', 'V3', 'V4', 'V5'], axis = 1, inplace = True)

# Count of each word
count = data.loc[:, :].sum()

# Popular items
pop_item = count.sort_values(0, ascending = False).head(5)

# Converting to Data frame
pop_item = pd.DataFrame(pop_item)

# Resetting index
pop_item = pop_item.reset_index()

# Renaming 
pop_item.rename({'index' : 'movie', 0 : 'count'}, axis = 1, inplace = True)

# Plot
ax = pop_item.plot.barh(x = 'movie', y = 'count')
plt.gca().invert_yaxis()
plt.title('Most Popular items')
plt.show()

# Frequent items
frequent_items = apriori(data, min_support = 0.007, max_len = 4, use_colnames = True)
frequent_items

# SOrting
frequent_items.sort_values('support', ascending = False, inplace = True)

# Association rules
rules = association_rules(frequent_items, metric = 'lift', min_threshold = 1)

# User defined function
def to_list(i):
    return sorted(list(i))

# Sorting on antecedents and consequensts
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

# Sorting
ma_X = sorted(ma_X)

# Converting it into list
rules_set = list(ma_X)

# Unique rules
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_set)]

# Empty list
index_rules = []

# pulling index values to empty list
for i in unique_rules_sets:
    index_rules.append(rules_set.index(i))
    
index_rules

# Redundancy
rules_no_redundancy = rules.iloc[index_rules, :]
rules_no_redundancy

# Top 10 rules
rules10 = rules_no_redundancy.sort_values('lift', ascending = False).head(10)
rules10

rules10.plot(x = "support", y = "confidence", c = rules10.lift, kind="scatter", s = 12, cmap = plt.cm.coolwarm)

###################################################################################################
3. A Mobile Phone manufacturing company wants to launch its three-brand new phone into the market, 
but before going with its traditional marketing approach this time it wants to analyze the data of 
its previous model sales in different regions and you have been hired as an Data Scientist to help 
them out, use the Association rules concept and provide your insights to the company’s marketing 
team to improve its sales.

####################################################################################################

# Imporing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Reading the data into Python
data = pd.read_csv("D:/Hands on/12_Association Rules/Assignment/myphonedata.csv")

# Droping the columns
data.drop(['V1', 'V2', 'V3'], axis = 1, inplace = True)

# Count of each word
count = data.loc[:, :].sum()

# TOp 5 words
pop_item = count.sort_values(0, ascending = False).head(5)

# Creating a data frame
pop_item = pd.DataFrame(pop_item)

# Resetting a index
pop_item = pop_item.reset_index()

# Renaming
pop_item.rename({'index' : 'movie', 0 : 'count'}, axis = 1, inplace = True)

# Graph
ax = pop_item.plot.barh(x = 'movie', y = 'count')
plt.gca().invert_yaxis()
plt.title('Most Popular items')
plt.show()

# Frequent items
frequent_items = apriori(data, min_support = 0.007, max_len = 4, use_colnames = True)
frequent_items

# SOriting
frequent_items.sort_values('support', ascending = False, inplace = True)

# Association rules
rules = association_rules(frequent_items, metric = 'lift', min_threshold = 1)

# Funciton
def to_list(i):
    return sorted(list(i))

# Sorting
ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

# Sorting
ma_X = sorted(ma_X)

# Converting it to list
rules_set = list(ma_X)

# Unique rules
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_set)]

# Empty list
index_rules = []

# appending index values to list
for i in unique_rules_sets:
    index_rules.append(rules_set.index(i))
    
index_rules

# Redundancy
rules_no_redundancy = rules.iloc[index_rules, :]
rules_no_redundancy

# Top 10 rules
rules10 = rules_no_redundancy.sort_values('lift', ascending = False).head(10)
rules10

rules10.plot(x = "support", y = "confidence", c = rules10.lift, kind="scatter", s = 12, cmap = plt.cm.coolwarm)

###################################################################################################

Q4. A retail store in India, has its transaction data, and it would like to know the buying pattern of the 
consumers in its locality, you have been assigned this task to provide the manager with rules 
on how the placement of products needs to be there in shelves so that it can improve the buying
patterns of consumes and increase customer footfall. 

###################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

transactions_retail1  = []

with open(r"D:/Hands on/12_Association Rules/Assignment/transactions_retail.csv") as f:
    transactions_retail1  = f.read()


transactions_retail1  = transactions_retail1.split("\n")

viewing = pd.DataFrame(transactions_retail1_list)

transactions_retail1_list = []

for i in transactions_retail1:
    transactions_retail1_list.append(i.split(","))
                          
all_transactions_retail1_list = [i for item in transactions_retail1_list for i in item]

from collections import Counter # ,OrderedDict

item_frequencies = Counter(all_transactions_retail1_list)

item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

frequtent_transactions = apriori(viewing, min_support = 0.05, max_len = 4, use_colnames = True)

