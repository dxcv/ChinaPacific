import numpy as np

c= np.diag([1,2,2])
d= np.ones([3,3])
e= np.dot( c,d )
import pandas as pd
import sklearn as sk
import statsmodels.api as sm
import matplotlib.pyplot as plt

e= pd.DataFrame( e,
                index= ['a', 'b', 'c'],
                columns= ['A', 'B', 'C'])

z= np.ones([9,9])
u= np.diag([3,4,5,6])
a= pd.DataFrame(z, index= None, columns= None )


print( e)

d= np.dot(np.ones([2,2]), np.diag([1,1]))

print(d)

sk.covariance.ledoit_wolf(X)
