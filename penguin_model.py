import pandas as pd

df = pd.read_csv('D:/Data Sets/penguins_size.csv')

df.dropna(inplace=True) #removing Null values
df.drop('island', axis=1, inplace=True)
#label encoding
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = enc.fit_transform(df[col])


#train test split
from sklearn.model_selection import train_test_split
y = df.species
df.drop('species', axis=1,inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df, y,test_size=0.15)


#model train
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_test)

print(f'The accuracy of model is {acc}')


#save model
from joblib import dump
dump(model,'penguin_model')