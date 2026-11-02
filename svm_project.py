import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.datasets import make_blobs


# 1. Load dataset from UCI

heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features.copy()
y = heart_disease.data.targets.copy()

print("Dataset information:")
print(heart_disease.metadata)

print("\nVariables:")
print(heart_disease.variables)

print("\nFirst five rows:")
print(X.head())

print("\nDataset shape:", X.shape)


# 2. Prepare the data

X.columns = X.columns.astype(str).str.strip().str.lower()
X = X.apply(pd.to_numeric, errors="coerce")

# The target column in the UCI dataset is num
y = y.iloc[:, 0]
y = pd.to_numeric(y, errors="coerce")

# Remove rows with missing target values
valid_rows = y.notna()
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# 0 = no disease, values above 0 = disease
y = (y > 0).astype(int)

# Remove features that contain only missing values
X = X.dropna(axis=1, how="all")

print("\nMissing values:")
print(X.isnull().sum())

print("\nClass distribution:")
print(y.value_counts())


# 3. Visualize class distribution

y.value_counts().sort_index().plot(
    kind="bar",
    color=["steelblue", "tomato"]
)

plt.title("Heart Disease Class Distribution")
plt.xlabel("Class (0 = No Disease, 1 = Disease)")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()


# 4. Split data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Build the SVM pipeline

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svm", SVC())
])


# 6. Find the best kernel and parameters

param_grid = [
    {
        "svm__kernel": ["linear"],
        "svm__C": [0.1, 1, 10]
    },
    {
        "svm__kernel": ["rbf"],
        "svm__C": [0.1, 1, 10],
        "svm__gamma": ["scale", 0.01, 0.1]
    },
    {
        "svm__kernel": ["poly"],
        "svm__C": [0.1, 1, 10],
        "svm__degree": [2, 3]
    }
]

cv_folds = min(5, int(y_train.value_counts().min()))

search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=cv_folds,
    n_jobs=-1
)

search.fit(X_train, y_train)

model = search.best_estimator_

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation F1 score:")
print(search.best_score_)


# 7. Evaluate the model

y_pred = model.predict(X_test)

print("\nModel evaluation")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1 Score:", round(f1_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 8. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# 9. Support Vectors

svm = model.named_steps["svm"]

print("\nBest kernel:", svm.kernel)
print("Support vectors per class:", svm.n_support_)
print("Total support vectors:", len(svm.support_vectors_))


# 10. Decision Boundary using two features

features = list(X.columns[:2])

imputer_2d = SimpleImputer(strategy="median")
scaler_2d = StandardScaler()

X_train_2d = imputer_2d.fit_transform(X_train[features])
X_train_2d = scaler_2d.fit_transform(X_train_2d)


def plot_boundary(ax, classifier, title):
    x_min = X_train_2d[:, 0].min() - 1
    x_max = X_train_2d[:, 0].max() + 1
    y_min = X_train_2d[:, 1].min() - 1
    y_max = X_train_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = classifier.predict(grid).reshape(xx.shape)

    ax.contourf(
        xx, yy, predictions,
        alpha=0.2,
        cmap="coolwarm"
    )

    scores = classifier.decision_function(grid).reshape(xx.shape)

    ax.contour(
        xx, yy, scores,
        levels=[0],
        colors="black",
        linewidths=2
    )

    ax.contour(
        xx, yy, scores,
        levels=[-1, 1],
        colors="gray",
        linestyles="dashed"
    )

    ax.scatter(
        X_train_2d[:, 0],
        X_train_2d[:, 1],
        c=y_train,
        cmap="coolwarm",
        edgecolors="black",
        s=30
    )

    ax.scatter(
        classifier.support_vectors_[:, 0],
        classifier.support_vectors_[:, 1],
        s=120,
        facecolors="none",
        edgecolors="gold",
        linewidths=2
    )

    ax.set_title(title)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])


# 11. Compare Linear, RBF and Polynomial kernels

kernels = [
    ("Linear", SVC(kernel="linear", C=1)),
    ("RBF", SVC(kernel="rbf", C=1, gamma="scale")),
    ("Polynomial", SVC(kernel="poly", C=1, degree=3))
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, classifier) in zip(axes, kernels):
    classifier.fit(X_train_2d, y_train)
    plot_boundary(ax, classifier, name + " Kernel")

plt.tight_layout()
plt.show()


# 12. Effect of C on the margin

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, c_value in zip(axes, [0.1, 100]):
    classifier = SVC(kernel="linear", C=c_value)
    classifier.fit(X_train_2d, y_train)

    plot_boundary(
        ax,
        classifier,
        "Linear SVM, C = " + str(c_value)
    )

plt.tight_layout()
plt.show()


# 13. Effect of Gamma on RBF

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, gamma_value in zip(axes, [0.01, 0.1, 10]):
    classifier = SVC(
        kernel="rbf",
        C=1,
        gamma=gamma_value
    )

    classifier.fit(X_train_2d, y_train)

    plot_boundary(
        ax,
        classifier,
        "RBF, gamma = " + str(gamma_value)
    )

plt.tight_layout()
plt.show()


# 14. Soft Margin and Large C demonstration

toy_X, toy_y = make_blobs(
    n_samples=100,
    centers=[(-2, -2), (2, 2)],
    cluster_std=1.2,
    random_state=42
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, c_value in zip(axes, [0.1, 10000]):
    classifier = SVC(kernel="linear", C=c_value)
    classifier.fit(toy_X, toy_y)

    xx, yy = np.meshgrid(
        np.linspace(toy_X[:, 0].min() - 1,
                    toy_X[:, 0].max() + 1, 200),
        np.linspace(toy_X[:, 1].min() - 1,
                    toy_X[:, 1].max() + 1, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = classifier.decision_function(grid).reshape(xx.shape)

    ax.scatter(
        toy_X[:, 0], toy_X[:, 1],
        c=toy_y, cmap="coolwarm"
    )

    ax.contour(
        xx, yy, scores,
        levels=[-1, 0, 1],
        colors=["gray", "black", "gray"],
        linestyles=["dashed", "solid", "dashed"]
    )

    ax.scatter(
        classifier.support_vectors_[:, 0],
        classifier.support_vectors_[:, 1],
        s=100,
        facecolors="none",
        edgecolors="gold"
    )

    ax.set_title("C = " + str(c_value))

plt.tight_layout()
plt.show()


# 15. Save the trained model

joblib.dump(model, "svm_heart_model.joblib")
joblib.dump(list(X.columns), "svm_feature_columns.joblib")

print("\nModel saved successfully!")
print("All SVM experiments completed.")