# Convert selected features to their human-readable format using BioMart
server = BiomartServer("http://www.ensembl.org/biomart")
dataset = server.datasets['hsapiens_gene_ensembl']

response = dataset.search({
    'filters': {'ensembl_gene_id': selected_features},  # Use selected features from models
    'attributes': ['ensembl_gene_id', 'external_gene_name']
})

# Parse response to create a mapping of Ensembl ID → Human Gene Name
biomart_results = response.text.strip().split("\n")
ensembl_to_symbol = {}
for line in biomart_results:
    ensembl_id, gene_symbol = line.split("\t")
    ensembl_to_symbol[ensembl_id] = gene_symbol if gene_symbol else "Unknown"

# Function to map features to human-readable gene symbols
def get_human_readable_features(feature_importance_df, top_n=10):
    top_features = feature_importance_df.head(top_n)
    human_readable_features = [
        (feature, ensembl_to_symbol.get(feature, "Unknown")) for feature in top_features['feature']
    ]
    return human_readable_features

# Get top 10 features with human-readable names
top_10_rf = get_human_readable_features(rf_feature_importance, top_n=10)
top_10_svm = get_human_readable_features(svm_feature_importance, top_n=10)

# Find common features in human-readable format
common_features_human_readable = [
    (feature, ensembl_to_symbol.get(feature, "Unknown")) for feature in common_features
]

# Print Results
print("\n🔹 Top 10 Biomarkers from Random Forest:")
for idx, (ensembl_id, gene_symbol) in enumerate(top_10_rf, start=1):
    print(f"{idx}. {gene_symbol} ({ensembl_id})")

print("\n🔹 Top 10 Biomarkers from SVM:")
for idx, (ensembl_id, gene_symbol) in enumerate(top_10_svm, start=1):
    print(f"{idx}. {gene_symbol} ({ensembl_id})")

print("\n🔹 Common Biomarkers in RF and SVM:")
if common_features_human_readable:
    for idx, (ensembl_id, gene_symbol) in enumerate(common_features_human_readable, start=1):
        print(f"{idx}. {gene_symbol} ({ensembl_id})")
else:
    print("No common top features found between RF and SVM.")
