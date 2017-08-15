####This is a Data Science Project
##### We are working on the Yelp Dataset


### importing third party libraries

import pandas 
import numpy
#from mylibrary import dataframe1



#=======Joining 14 csv files to make a combined csv file=========
'''
colNames = ['business_id',
 'full_address',
 'hours/Friday/close',
 'hours/Friday/open',
 'hours/Tuesday/close',
 'hours/Tuesday/open',
 'hours/Thursday/close',
 'hours/Thursday/open',
 'hours/Wednesday/close',
 'hours/Wednesday/open',
 'hours/Monday/close',
 'hours/Monday/open',
 'open',
 'categories/0',
 'categories/1',
 'city',
 'review_count',
 'name',
 'longitude',
 'state',
 'stars',
 'latitude',
 'attributes/Take-out',
 'attributes/Drive-Thru',
 'attributes/Good For/dessert',
 'attributes/Good For/latenight',
 'attributes/Good For/lunch',
 'attributes/Good For/dinner',
 'attributes/Good For/brunch',
 'attributes/Good For/breakfast',
 'attributes/Caters',
 'attributes/Noise Level',
 'attributes/Takes Reservations',
 'attributes/Delivery',
 'attributes/Ambience/romantic',
 'attributes/Ambience/intimate',
 'attributes/Ambience/classy',
 'attributes/Ambience/hipster',
 'attributes/Ambience/divey',
 'attributes/Ambience/touristy',
 'attributes/Ambience/trendy',
 'attributes/Ambience/upscale',
 'attributes/Ambience/casual',
 'attributes/Parking/garage',
 'attributes/Parking/street',
 'attributes/Parking/validated',
 'attributes/Parking/lot',
 'attributes/Parking/valet',
 'attributes/Has TV',
 'attributes/Outdoor Seating',
 'attributes/Attire',
 'attributes/Alcohol',
 'attributes/Waiter Service',
 'attributes/Accepts Credit Cards',
 'attributes/Good for Kids',
 'attributes/Good For Groups',
 'attributes/Price Range',
 'type',
 'attributes/Happy Hour',
 'categories/2',
 'hours/Sunday/close',
 'hours/Sunday/open',
 'hours/Saturday/close',
 'hours/Saturday/open',
 'categories/3',
 'categories/4',
 'categories/5',
 'attributes/Good For Dancing',
 'attributes/Coat Check',
 'attributes/Smoking',
 'attributes/Wi-Fi',
 'attributes/Music/dj',
 'neighborhoods/0',
 'attributes/Wheelchair Accessible',
 'attributes/Dogs Allowed',
 'attributes/BYOB',
 'attributes/Corkage',
 'attributes/BYOB/Corkage',
 'attributes/Order at Counter',
 'attributes/Music/background_music',
 'attributes/Music/jukebox',
 'attributes/Music/live',
 'attributes/Music/video',
 'attributes/Music/karaoke',
 'attributes/By Appointment Only',
 'categories/6',
 'attributes/Open 24 Hours',
 'neighborhoods/1',
 'attributes/Hair Types Specialized In/coloring',
 'attributes/Hair Types Specialized In/africanamerican',
 'attributes/Hair Types Specialized In/curly',
 'attributes/Hair Types Specialized In/perms',
 'attributes/Hair Types Specialized In/kids',
 'attributes/Hair Types Specialized In/extensions',
 'attributes/Hair Types Specialized In/asian',
 'attributes/Hair Types Specialized In/straightperms',
 'attributes/Accepts Insurance',
 'categories/7',
 'attributes/Ages Allowed',
 'attributes/Dietary Restrictions/dairy-free',
 'attributes/Dietary Restrictions/gluten-free',
 'attributes/Dietary Restrictions/vegan',
 'attributes/Dietary Restrictions/kosher',
 'attributes/Dietary Restrictions/halal',
 'attributes/Dietary Restrictions/soy-free',
 'attributes/Dietary Restrictions/vegetarian',
 'categories/8',
 'neighborhoods/2',
 'categories/9']

df = pandas.DataFrame(columns = colNames)
n = len(df)


for i in range(13,15):
    fileName = 'CSV '+str(i)+'.csv'
    #fileName = 'C:\\Users\\ketan walia\\Desktop\\Pyhton Yelp Project\\CSV '+str(i)+'.csv'
    csvNew = pandas.read_csv(fileName)
    m = len(csvNew)
    df1 = pandas.DataFrame(csvNew, index = range(n,n+m))
    frames = [df,df1]
    df = pandas.concat(frames)
    


df.to_csv("CombinedCSV.csv")

#========Filtering Data for open Restaurants===========

file = pandas.read_csv('CombinedCSV.csv')


colNames = list(file.columns.values)
df = pandas.DataFrame(columns = colNames)

n = len(file)

count = 0

for i in range(n):
    if (file['open'][i] == True):
        
        if(file['categories/0'][i] == 'Restaurants'):
            df.loc[count] = file.loc[i]
            count += 1
        elif(file['categories/1'][i] == 'Restaurants'):
            df.loc[count] = file.loc[i]
            count += 1
        elif(file['categories/2'][i] == 'Restaurants'):
            df.loc[count] = file.loc[i]
            count += 1
        elif(file['categories/3'][i] == 'Restaurants'):
            df.loc[count] = file.loc[i]
            count += 1
        elif(file['categories/4'][i] == 'Restaurants'):
            df.loc[count] = file.loc[i]
            count += 1
        elif(file['categories/5'][i] == 'Restaurants'):
            df.loc[count] = file.loc[i]
            count += 1
        elif(file['categories/6'][i] == 'Restaurants'):
            df.loc[count] = file.loc[i]
            count += 1
        elif(file['categories/7'][i] == 'Restaurants'):
            df.loc[count] = file.loc[i]
            count += 1
        elif(file['categories/8'][i] == 'Restaurants'):
            df.loc[count] = file.loc[i]
            count += 1
        elif(file['categories/9'][i] == 'Restaurants'):
            df.loc[count] = file.loc[i]
            count += 1

df.to_csv("OpenRestaurants.csv")


#============Create regressor variable============

file=pandas.read_csv("OpenRestaurants.csv")

maxCount = max(file['review_count'])
minCount = min(file['review_count'])
diffCount = maxCount - minCount

newCount = (file['review_count'] - minCount)/diffCount

maxStars = max(file['stars'])
minStars = min(file['stars'])
diffStars = maxStars - minStars

newStars = (file['stars'] - minStars)/diffStars

popularityIndex = newCount * newStars

file['newCount'] = newCount
file['newStars'] = newStars
file['popularityIndex'] = popularityIndex

file.to_csv("PopularityIndexRestaurants.csv")


#========Creating Column Stats=============

file = pandas.read_csv("PopularityIndexRestaurants.csv")

file = file.drop(file.columns[[0,1,2]], axis = 1)

colNames = list(file.columns.values)


total = len(file)
cols = len(file.columns)

colStats = pandas.DataFrame(columns = ['ColName','Missing','False','True','Populated','Total'])

for i in range(cols):
    blanks = file[colNames[i]].isnull().sum()
    populated = file[colNames[i]].count()
    temp = list(file[colNames[i]])
    trues = temp.count(True)
    falses = temp.count(False)
    total = blanks + populated
    PercentPopulated=(populated/total)*100
    myData = [{'ColName':colNames[i],'Missing':blanks,'False':falses,'True':trues
               ,'Populated':populated, 'Total':total,'PercentPopulated':PercentPopulated}]
    colStats = colStats.append(myData,ignore_index = True)

    
colStats.to_csv("ColumnStats.csv")





#==========Selecting Attributes (over 50% populated attributes)===============



selectedCols = pandas.DataFrame(columns = ['ColName','Missing','False','True','Populated','Total','PercentPopulated'])



for i in range(cols):
    if (colStats['PercentPopulated'][i] > 50):
        selectedCols = selectedCols.append(colStats.loc[i,:])


selCols = list(selectedCols['ColName'])

selAttributesFile = file[selCols]

selAttributesFile.to_csv('SelectAttributesFile.csv')
'''

#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#

#=============================================================================#

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib as plt



raw_data=pd.read_csv('C:\\Users\\ketan walia\\Desktop\\SelectAttributesFile1314.csv')
#raw_data=pd.read_csv('SelectAttributesFile.csv')
raw_data.apply(lambda x:pd.to_numeric(x,errors='coerce'))

data=pd.DataFrame(raw_data)

data.drop(data.columns[[0,2,12,24,31,35,36,37,38,39,40,41,58,59,62,63]], inplace=True,axis=1)

feature_cols = list(data.columns[0:50])

target_col = data.columns[-1]

y_all=data[target_col]

X_all = data[feature_cols]


def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace([True, False], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX


################### Making Classes for variable description#################


from mylibrary import billu, yelp
import pandas
import numpy
import matplotlib as plt
import matplotlib.pyplot as plt

Regressor_Var = billu(y_all)

Reg_obj=yelp(y_all)

print("Regressor Variable Variance:",Regressor_Var.variance())

print("Regressor Variable Frequency:",Regressor_Var.frequency())

print("Regressor Variable Maximum Value:",Regressor_Var.max_value())

print("Regressor Variable Minimum Value:",Regressor_Var.min_value())

print("-----------------------------------------")

print("Regressor Variable Mean Value:",Reg_obj.col_average())

#print("Deleting the last item from the regressor variable")
#Reg_obj.del_item()

#print("Regressor mean:",Regressor_av.average())

############################ Fitting the Linear regression Model##########################


X_all = preprocess_features(X_all)
#print(X_all.describe())
#plt.hist(y_all)
#plt.xlabel('Popularity Index', fontsize=18)
#plt.show()
#print (type(X_all))

y_all.fillna(0)
X_all=X_all.fillna(0)

model=LinearRegression()

model.fit(X_all,y_all)

print("Linear Regression Model Intercept:",model.intercept_)

print("Number of coefficients built in by the Linear Regression Model:",model.coef_.shape)

#print(model.coef_)

model.fit(X_all[::2],y_all[::2]) #Training on the test set

print("R2 score: %s" % model.score(X_all[1::2],y_all[1::2]))


########################### Fitting the Ridge Regression Model########################

from sklearn.linear_model import Ridge

model2=Ridge(alpha=0.1)

model2.fit(X_all,y_all)

print("Linear Regression Model Intercept:",model2.intercept_)

print("Number of coefficients built in by the Linear Regression Model:",model2.coef_.shape)

#print(model2.coef_)

model2.fit(X_all[::2],y_all[::2]) # Training on the test set

print("R2 score: %s" % model2.score(X_all[1::2],y_all[1::2]))

#Regressor_Var.distribution()

#print(X_all.describe())
