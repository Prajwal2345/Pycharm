import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('iris.csv')
print(df.head())

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['Class']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=1)

sc = StandardScaler()

Xtrain_std = sc.fit_transform(Xtrain)

Xtest_std = sc.transform(Xtest)

classifier = RandomForestClassifier()

classifier.fit(Xtrain_std, ytrain)

pickle.dump(classifier, open("model.pkl", "wb"))
