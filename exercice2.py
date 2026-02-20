import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#chargeons le dataset iris et explorons les données
iris = load_iris()
print(iris.keys()) #pour recuperer les clés de IRIS 
print(f"Type de iris.data: {type(iris.data)}")
print(f"Type de iris.target: {type(iris.target)}")
print(f"la taille du data: {iris.data.shape}") #dimension du data
print(f"la taille du target: {iris.target.shape}") #dimension du target
#examinons les premieres lignes des données et de label
print(f"les 5 premières lignes des données: \n {iris.data[:5]}") #Appercu du 5 premieres lignes du donnée
print(f"les 5 premieres lignes du target: \n {iris.target[:5]}") #Apercu du 5 premieres lignes du target

#Visualisation des données
#le diagrame de dispersion
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Espece"] = iris.target 
print(df)
plt.scatter(df[df["Espece"]==0]["sepal length (cm)"], df[df["Espece"]==0]["sepal width (cm)"], color="red", label="Setosa")
plt.scatter(df[df["Espece"]==1]["sepal length (cm)"], df[df["Espece"]==1]["sepal width (cm)"], color="blue", label="Versicolor")
plt.scatter(df[df["Espece"]==2]["sepal length (cm)"], df[df["Espece"]==2]["sepal width (cm)"], color="green", label="Virginica")
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.legend()
plt.title("Dataset Iris")
plt.show()

#Preparation des données
x = df.drop(columns=["Espece"])
y = df["Espece"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
print(f"x.shape:{x.shape}, x_train.shape:{x_train.shape}, x_test:{x_test.shape} \n")
print(f"y.shape:{y.shape}, y_train.shape:{y_train.shape}, y_test:{y_test.shape} \n" )

#Entrainons le model, effectuons de prediction
modele = LogisticRegression()
modele.fit(x_train, y_train)

y_predict = modele.predict(x_test)
print(f"Prediction: {y_predict} \n")
print(f"y_test: {y_test}")

#Evaluons le modele 
print(f"La precision du modele est de: {accuracy_score(y_predict, y_test)*100}%")
print(f"{classification_report(y_predict, y_test, target_names=iris.target_names)}")

score = modele.score(x_train, y_train)
print(f"score: {score*100:.2f}%")