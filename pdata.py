import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import graphlab
from graphlab import SFrame
from graphlab.toolkits.recommender import ranking_factorization_recommender

col_names = ["user_id", "item_id", "rating", "timestamp"]
data = pd.read_table('./u.data', names=col_names)
data = data.drop('timestamp', 1)
data.info()
pdf = pd.DataFrame(data)

#SFrame of whole data
sf = SFrame(pdf)

train, test = train_test_split(pdf, test_size = 0.3)
#training,validation = train_test_split(train, test_size = 0.25)

testsf = SFrame(test)
trainsf=SFrame(train)


"""
PopularityRecommender methods
"""

#myrating = ranking_factorization_recommender.create(sf, target='rating')
itemrating = graphlab.item_similarity_recommender.create(sf,target='rating', similarity_type='cosine')


Number_Ratings = len(data)
Number_Movies = len(np.unique(data['item_id']))
Number_Users = len(np.unique(data['user_id']))
Sparcity = (Number_Ratings/(Number_Movies*Number_Users))*100
print('Number_Users:',Number_Users, 'Number_Movies:',Number_Movies, 'Number_Ratings:',Number_Ratings)
print('Sparcity(%)', Sparcity)

#print(pdf)
#print(len(train), len(test), len(training), len(validation))

m = graphlab.recommender.create(trainsf, target='rating')
print(m.evaluate_rmse(testsf, target='rating') )

#print(myrating)

print(itemrating.recommend() )

plt.hist(data['rating'])
plt.show()
plt.savefig('myfig')