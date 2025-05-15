import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

data = pd.read_csv('Superstore Sales Dataset.csv')

features = ['Sales']  
target = 'Category'

available_features = []
for col in features:
    if col in data.columns:
        available_features.append(col)

if not available_features:
    available_features = ['Sales']  

le = LabelEncoder()
data[target] = le.fit_transform(data[target])

X = data[available_features]
y = data[target]

X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Precision del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificacion:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
