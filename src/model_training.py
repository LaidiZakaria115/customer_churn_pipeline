from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)
from data_preprocessing import preprocess
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(model , X_train , X_test , y_train , y_test):
    # Cross val to see whether its consistent and generalizes well
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print("Cross-validation F1 scores:", cv_scores)
    print("Mean CV F1-score:", np.mean(cv_scores))

    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # needed for ROC-AUC

    # Key metrics
    print("\n--- Model Evaluation ---")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    # Classification report (pretty summary)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title("ROC Curve")
    plt.show()




df = preprocess()

# Splitting features from target
X = df.drop(["Churn" , "customerID"] , axis = 1)
X = X.select_dtypes(include=["int64" , "float64"])
y = df["Churn"]

# Train/Test Split
X_train, X_test , y_train , y_test = train_test_split(
    X , y , test_size=0.2 , random_state=42)
print(X_train.dtypes)
# Fix Imbalance by SMOTE ovesampling the minority class
smote = SMOTE(sampling_strategy="auto" , random_state=42)
X_smote , y_smote = smote.fit_resample(X_train , y_train)


# Train the model using RandomForest for now to compare later to LightGBM

model_RFC = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)


train_data = lgb.Dataset(X_smote , label = y_smote)
valid_data = lgb.Dataset(X_test , label=y_test)

model_LGBM = lgb.LGBMClassifier(
    boosting_type="gbdt",
    num_leaves=50,
    max_depth=6,
    n_estimators=500,
    learning_rate=0.01,
    random_state=42,
    colsample_bytree=0.7,
    subsample=0.7
)

model_LGBM.fit(X_smote , y_smote)



print("LightGBM evaluation :\n", evaluate_model(model_LGBM , X_smote , X_test , y_smote , y_test))






 
