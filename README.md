# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step-by-Step Explanation

1. Import Required Libraries

2. Load and View the Dataset

3. Drop Unnecessary Columns

4. Convert Categorical Columns to category Type

5. Convert Categories to Numeric Codes

6. Separate Features (X) and Target (y)

7. Initialize Model Parameters (Weights)

8. Define the Sigmoid Function

9. Define the Loss (Cost) Function

10. Implement Gradient Descent

11. Define Prediction Function

12. Make Predictions and Compute Accuracy

13. Predict for New Students


## Program:
```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
iris=load_iris()
df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target
print(df.head())
X = df.drop('target',axis=1)
y=df['target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
```
[or]
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.calibration import CalibratedClassifierCV  
import warnings
warnings.filterwarnings("ignore")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target) 
print("Dataset:", data.DESCR.splitlines()[0])
print("X shape:", X.shape, "y shape:", y.shape)
print()
print("First 5 rows:")
print(X.iloc[:, :6].head())

feat1, feat2 = "mean radius", "mean texture"
X_vis = X[[feat1, feat2]].values


X_full = X.values

X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.25, random_state=42, stratify=y)
Xv_train, Xv_test, yv_train, yv_test = train_test_split(X_vis, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

scaler_vis = StandardScaler().fit(Xv_train)
Xv_train_s = scaler_vis.transform(Xv_train)
Xv_test_s = scaler_vis.transform(Xv_test)

clf = SGDClassifier(
    loss='log_loss',           
    penalty='l2',
    alpha=1e-4,
    max_iter=1000,
    tol=1e-4,
    learning_rate='optimal',  
    random_state=42,
    verbose=0
)

clf.fit(X_train_s, y_train)


calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv=5)
calibrated.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)
y_proba = calibrated.predict_proba(X_test_s)[:, 1]  

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Test set metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print()

cv_scores = cross_val_score(clf, scaler.transform(X_full), y, cv=5, scoring='accuracy')
print("5-fold CV accuracy: mean={:.4f} std={:.4f}".format(cv_scores.mean(), cv_scores.std()))
print()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax)
ax.set_title("Confusion Matrix (Test set)")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6,5))
RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="SGD Logistic (calibrated)").plot(ax=ax)
ax.set_title(f"ROC Curve (AUC = {roc_auc:.3f})")
plt.show()

clf_vis = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-4, random_state=42)
clf_vis.fit(Xv_train_s, yv_train)

# Make mesh
xx_min, xx_max = Xv_train_s[:, 0].min() - 1, Xv_train_s[:, 0].max() + 1
yy_min, yy_max = Xv_train_s[:, 1].min() - 1, Xv_train_s[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(xx_min, xx_max, 300), np.linspace(yy_min, yy_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

Z = clf_vis.predict(grid).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7,6))
ax.contourf(xx, yy, Z, alpha=0.2)
# plot training points
ax.scatter(Xv_train_s[:, 0][yv_train==0], Xv_train_s[:, 1][yv_train==0], marker='o', label=data.target_names[0], edgecolor='k')
ax.scatter(Xv_train_s[:, 0][yv_train==1], Xv_train_s[:, 1][yv_train==1], marker='^', label=data.target_names[1], edgecolor='k')
ax.set_xlabel(feat1 + " (scaled)")
ax.set_ylabel(feat2 + " (scaled)")
ax.set_title("Decision boundary (SGD Logistic) — trained on 2 features")
ax.legend()
plt.show()

/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: MATHIYAZHAGAN A
RegisterNumber:  212225230170
*/
```

## Output:

<img width="822" height="405" alt="Screenshot 2026-08-08 104515" src="https://github.com/user-attachments/assets/a2c9c72d-626f-4f39-9449-535080d5fc2e" />
<img width="746" height="496" alt="Screenshot 2026-08-08 104523" src="https://github.com/user-attachments/assets/b257bf1f-5155-475d-81e9-cb7a89565f14" />

[or]

<img width="747" height="227" alt="Screenshot 2026-08-08 105218" src="https://github.com/user-attachments/assets/51ad00eb-43e6-43f4-b247-5c44b2bb8132" />
<img width="546" height="316" alt="Screenshot 2026-08-08 105227" src="https://github.com/user-attachments/assets/7c495dcd-5e5f-4aab-b6cd-c865a84dd6b3" />
<img width="489" height="391" alt="1" src="https://github.com/user-attachments/assets/ee57c493-2be1-4486-8fb3-cd82c2272449" />
<img width="536" height="468" alt="2" src="https://github.com/user-attachments/assets/1a2d0c6a-4bb3-40c0-8435-b0cee6b372c1" />
<img width="616" height="545" alt="3" src="https://github.com/user-attachments/assets/dc30af22-9ad5-4198-b248-49a3d03f6ffc" />




## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
