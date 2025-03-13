import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc

# Load dataset
data = pd.read_csv("/Users/saieshprabhu/Desktop/Rsem_count.csv")
print(data.shape[0])
print(data.shape[1])

# Preprocess data
X = data.iloc[:, 1:-1]  # Exclude Sample_Name and Group columns
y = data.iloc[:, -1]  # Target column

# Ensure y contains binary labels (0 and 1)
y = np.where(y == y.min(), 0, 1)

# Feature selection
selector = SelectKBest(score_func=f_classif, k=170)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# Split data into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced)

# Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)

# SVM Classifier
svm_model = SVC(
    kernel='rbf',
    probability=True,
    class_weight='balanced',
    random_state=42
)

# Train models
rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Evaluate models
rf_predictions = rf_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]
svm_probabilities = svm_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))
print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))

# Compute ROC curve and AUC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probabilities, pos_label=1)
rf_auc = auc(rf_fpr, rf_tpr)

svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probabilities, pos_label=1)
svm_auc = auc(svm_fpr, svm_tpr)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Get feature importance from Random Forest
rf_feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features from Random Forest:")
print(rf_feature_importance.head(10))

# Get feature importance for SVM using permutation importance from sklearn
from sklearn.inspection import permutation_importance
svm_importance = permutation_importance(svm_model, X_test, y_test, scoring='accuracy', n_repeats=10, random_state=42)
svm_feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': svm_importance.importances_mean
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features from SVM:")
print(svm_feature_importance.head(10))

# Find common important features
common_features = set(rf_feature_importance['feature'].head(20)) & set(svm_feature_importance['feature'].head(20))
print("\nCommon Top Features in Random Forest and SVM:")
print(common_features)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
cv_rf = cross_val_score(rf_model, X_balanced, y_balanced, cv=loo, scoring='accuracy')
cv_svm = cross_val_score(svm_model, X_balanced, y_balanced, cv=loo, scoring='accuracy')

print("Random Forest LOOCV Accuracy:", np.mean(cv_rf))
print("SVM LOOCV Accuracy:", np.mean(cv_svm))

accuracies_rf = []
accuracies_svm = []

for _ in range(10):
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.3, random_state=None, stratify=y_balanced)
    
    rf_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    accuracies_rf.append(rf_model.score(X_test, y_test))
    accuracies_svm.append(svm_model.score(X_test, y_test))

print("RF Mean Accuracy:", np.mean(accuracies_rf), "±", np.std(accuracies_rf))
print("SVM Mean Accuracy:", np.mean(accuracies_svm), "±", np.std(accuracies_svm))

