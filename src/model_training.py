from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , f1_score , precision_score
from data_preprocessing import preprocess


df = preprocess()

# Splitting features from target
X = df.drop("churn" , axis = 1)
y = df["churn"]

# Train/Test Split
X_train, X_test , y_train , y_test = train_test_split(
    X,y,test_size=0.75,train_size=0.25 , random_state=42)

# Fix Imbalance by SMOTE ovesampling the minority class
smote = SMOTE(sampling_strategy="auto" , random_state=42)
X_smote , y_smote = smote.fit_resample(X_train , y_train)


# Train the model using RandomForest for now to compare later to LightGBM

model = RandomForestClassifier()
model.fit(X_smote , y_smote)


y_pred = model.predict(X_test)
print("Training Accuracy :" , model.score(X_smote , y_smote) )
print("Test Accuracy :" , accuracy_score(y_test , y_pred))




