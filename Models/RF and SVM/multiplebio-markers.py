from collections import Counter
from biomart import BiomartServer

# Store top features per model
top_features_per_model = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Get feature importance
    if model_name == "SVM":
        importance = np.abs(model.coef_).mean(axis=0)  # Use absolute coefficients
    else:
        importance = model.feature_importances_
    
    feature_importance_dict[model_name] = importance
    
    # Get top 10 features per model
    top_features_idx = np.argsort(importance)[-20:]
    top_features_per_model[model_name] = data.columns[1:-1][top_features_idx].tolist()

# Compute averaged feature importance
total_importance = np.mean([feature_importance_dict[m] for m in models.keys()], axis=0)
top_features_idx_avg = np.argsort(total_importance)[-10:]
top_features_avg = data.columns[1:-1][top_features_idx_avg].tolist()

# Count how often each feature appears in top 10 lists
all_top_features = []
for model, features in top_features_per_model.items():
    all_top_features.extend(features)

feature_counts = Counter(all_top_features)

# Find consensus features: Selected by at least 3 out of 4 models
consensus_features = [feature for feature, count in feature_counts.items() if count >= 3]
common_features_all_models = [feature for feature, count in feature_counts.items() if count == 4]  # Features in all 4 models

# 🔹 Connect to BioMart to Get Human-Readable Gene Names
server = BiomartServer("http://www.ensembl.org/biomart")
dataset = server.datasets['hsapiens_gene_ensembl']

# Retrieve gene symbols for selected Ensembl Gene IDs
response = dataset.search({
    'filters': {'ensembl_gene_id': list(set(all_top_features))},
    'attributes': ['ensembl_gene_id', 'external_gene_name']
})

# Parse BioMart response into a dictionary (Ensembl ID → Gene Symbol)
biomart_results = response.text.strip().split("\n")
ensembl_to_symbol = {line.split("\t")[0]: line.split("\t")[1] for line in biomart_results if "\t" in line}

# Function to print features in a formatted way
def print_features(title, features):
    print(f"\n🔹 {title}:")
    if features:
        for idx, feature in enumerate(features, start=1):
            gene_symbol = ensembl_to_symbol.get(feature, "Unknown")  # Get human-readable name
            print(f"{idx}. {gene_symbol} ({feature})")
    else:
        print("No features found.")

# Print results
print_features("Top 10 Features from Random Forest", top_features_per_model["Random Forest"])
print_features("Top 10 Features from SVM", top_features_per_model["SVM"])
print_features("Top 10 Features from XGBoost", top_features_per_model["XGBoost"])
print_features("Top 10 Features from LightGBM", top_features_per_model["LightGBM"])
print_features("Top 10 Features Averaged Across All Models", top_features_avg)
print_features("Consensus Features (Selected by at least 3 out of 4 models)", consensus_features)
print_features("Features Selected by All 4 Models", common_features_all_models)
