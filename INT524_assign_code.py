#import the dataset
import pandas as pd
import numpy as np
df = pd.read_csv('C:\\Users\\Birjit\\OneDrive\\Documents\\GitHub\\INT524_project\\assign\\dataset.csv')
print(df.head())
print(df.tail())
# No. of rows as Features
print(df.shape[0])
# No. of columns as samples
print(df.shape[1])
#any missing values
df.isnull().sum()
# Divide data into input and target
target=df['LastUpdated']
print(target.shape)
cols = df.columns
print(cols)
cols = df.columns[:3]
print(cols)
input_ = df[cols]
print(input_.shape)
cols1 = df.columns[3:]
print(cols1)
#Details of given data
input_.describe()
print(input_.isnull().sum())
print(input_.nunique())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(input_['SystemCodeNumber'].values)
print(input_)
input_ = y [:, np.newaxis]
import numpy as np
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input_,target,test_size = 0.2, random_state = 0)
print(np.shape(input_))
print(np.shape(x_train))
print(np.shape(x_test))
print(y_test)
print(y_test.shape)
df1 = df.iloc[:,[1,2]]
print(df1)

#1.  K-Means cluster
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
km = KMeans()
y_km = km.fit_predict(df1)
print(y_km)

plt.scatter(df1.iloc[y_km == 0,0], df1.iloc[y_km == 0,1])
plt.scatter(df1.iloc[y_km == 1,0], df1.iloc[y_km == 1,1])
plt.scatter(df1.iloc[y_km == 0,0], df1.iloc[y_km == 0,1])
plt.scatter(df1.iloc[y_km == 1,0], df1.iloc[y_km == 1,1])
plt.scatter(df1.iloc[y_km == 0,0], df1.iloc[y_km == 0,1])
plt.title('KMeans Clustering')
plt.plot()
plt.show()


#2. DBSCAN CLUSTERING
from sklearn.cluster import DBSCAN
db=DBSCAN(eps=0.2,min_samples=5,metric='euclidean')
y_db=db.fit_predict(df1)
plt.figure()
plt.scatter(df1.iloc[y_db==0,0],df1.iloc[y_db==0,1],c='lightblue',edgecolor='black', marker='o',s=40,label='cluster 1')
plt.scatter(df1.iloc[y_db==1,0],df1.iloc[y_db==1,1],c='red',edgecolor='black', marker='s',s=40,label='cluster 1')
plt.title('DBSCAN Clustering')
plt.legend()
plt.show()

#3. Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=3,affinity='euclidean', linkage='complete')
labels=ac.fit_predict(df1)
print('Cluster labels:%s'%labels)
import matplotlib.pyplot as plt
plt.scatter(df1.iloc[labels == 0,0], df1.iloc[labels == 0,1])
plt.scatter(df1.iloc[labels == 1,0], df1.iloc[labels == 1,1])
plt.scatter(df1.iloc[labels == 0,0], df1.iloc[labels == 0,1])
plt.scatter(df1.iloc[labels == 1,0], df1.iloc[labels == 1,1])
plt.scatter(df1.iloc[labels == 0,0], df1.iloc[labels == 0,1])
plt.title('Agglomerative Clustering')
plt.plot()
plt.show()





#1. Linear Regression

from sklearn.linear_model import LinearRegression
X = df[['Capacity']].values
Y = df[['Occupancy']].values
plt.scatter(X,Y)
plt.show()
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.30, random_state = 0)
slr = LinearRegression()
slr.fit(x_train,y_train)
y_predict = slr.predict(x_test)
plt.plot(x_test,y_predict, color = 'r')
plt.scatter(x_test,y_test,color ='g')
plt.title('linear regression')
plt.show()
slr.score(x_test,y_test)

from sklearn.metrics import mean_squared_error, r2_score
print(mean_squared_error(slr.predict(x_train),y_train))
print(r2_score(slr.predict(x_train),y_train))


#2. Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
def lin_regplot(x,y,model):
    plt.scatter(x,y,c = 'steelblue', edgecolor = 'white', s = 70)
    plt.plot(x,model.predict(x), color = 'black', lw = 2)
tree1 = DecisionTreeRegressor(max_depth = 3, random_state = 0)
tree1.fit(X,Y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], Y[sort_idx], tree1)
plt.xlabel('%lower status of population')
plt.ylabel('price in $100s')
plt.title('Decision Tree Regressor')
plt.show()
print(mean_squared_error(tree1.predict(x_train),y_train))
print(r2_score(tree1.predict(x_train),y_train))


#3. Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
tree2 = RandomForestRegressor(max_depth = 3, random_state = 0)
tree2.fit(X,Y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], Y[sort_idx], tree2)
plt.xlabel('%lower status of population')
plt.ylabel('price in $100s')
plt.title('Random Forest Regressor')
plt.show()
print(mean_squared_error(tree2.predict(x_train),y_train))
print(r2_score(tree2.predict(x_train),y_train))

# Create a function for the models
def classifications(X_train, Y_train):
        
#1.  Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, Y_train)
    
#2.  Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)
    
#3.  Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0, max_iter = 10000)
    log.fit(X_train, Y_train)
    
    
    # Print the models accuracy on the training data
   
    print('[1] Decision Tree Classifier Training Accuracy:', tree.score(X_train, Y_train))
    print('[2] Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
    print('[3] Logistic Regression Classifier Training Accuracy:', log.score(X_train, Y_train))
    return tree, forest, log

# Getting all the models
model = classifications(x_train, y_train)


