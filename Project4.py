import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("insurance_data.csv")
print(df.head())

x = df[["age"]]
y = df["bought_insurance"]

plt.scatter(x, y, color='red')
plt.xlabel("Age")
plt.ylabel("Yes / No")
plt.title("Insurance Purchase vs Age")
plt.show()

model = LogisticRegression()
model.fit(x, y)

A = int(input("Enter age: "))
output = model.predict([[A]])
print("Prediction (0 = No, 1 = Yes):", output[0])

import pickle
pickle.dump(model,open("insurancedata.pkl","wb"))
pickle_model = pickle.load(open("insurancedata.pkl","rb"))
print(pickle_model.predict([[A]]))

