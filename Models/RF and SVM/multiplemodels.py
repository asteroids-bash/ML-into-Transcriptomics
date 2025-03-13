import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN
from biomart import BiomartServer

# Set global random seed for consistency
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Load dataset
data = pd.read_csv("/Users/saieshprabhu/Desktop/Rsem_count.csv")

# Preprocess data
X = data.iloc[:, 1:-1]  # Exclude Sample_Name and Group columns
y = data.iloc[:, -1]    # Target column

# Ensure binary labels
y = np.where(y == y.min(), 0, 1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply ADASYN for class balancing (set shuffle=False to prevent variations)
adasyn = ADASYN(random_state=RANDOM_SEED, sampling_strategy='auto')
X_balanced, y_balanced = adasyn.fit_resample(X_scaled, y)

# Split into train-test (with fixed random_state)
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=RANDOM_SEED, stratify=y_balanced)

# Initialize models (set random_state)
models = {
    "XGBoost": xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_SEED),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "SVM": SVC(kernel='linear', probability=True, random_state=RANDOM_SEED)
}

feature_importance_dict = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    if model_name == "SVM":
        importance = np.abs(model.coef_).mean(axis=0)  # Use absolute coefficients for feature importance
    else:
        importance = model.feature_importances_
    
    feature_importance_dict[model_name] = importance

# Compute average feature importance across all models
total_importance = np.mean([feature_importance_dict[m] for m in models.keys()], axis=0)
top_features_idx = np.argsort(total_importance)[-10:]
selected_feature_names = data.columns[1:-1][top_features_idx]

# Connect to BioMart to get human-readable gene names
server = BiomartServer("http://www.ensembl.org/biomart")
dataset = server.datasets['hsapiens_gene_ensembl']

response = dataset.search({
    'filters': {'ensembl_gene_id': selected_feature_names.tolist()},
    'attributes': ['ensembl_gene_id', 'external_gene_name']
})

# Parse the response to create a dictionary mapping
biomart_results = response.text.strip().split("\n")
ensembl_to_symbol = {}
for line in biomart_results:
    ensembl_id, gene_symbol = line.split("\t")
    ensembl_to_symbol[ensembl_id] = gene_symbol

# Translate selected feature names to human-readable format
# Translate selected feature names to human-readable format along with Ensembl ID
human_readable_biomarkers = [(feature, ensembl_to_symbol.get(feature, "Unknown")) for feature in selected_feature_names]

# Print Top 10 Biomarkers in a human-readable format
print("Top 10 Biomarkers (Ensembl ID & Gene Symbol):")
for idx, (ensembl_id, biomarker) in enumerate(human_readable_biomarkers, start=1):
    print(f"{idx}. {biomarker} ({ensembl_id})")

