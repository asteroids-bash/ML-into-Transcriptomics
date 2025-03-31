

 🧬 ML into Transcriptomics: Biomarker Discovery using RF & SVM

This project applies machine learning techniques (Random Forest, SVM,LightGBM, XGBoost) to transcriptomic data for the identification of potential biomarkers. It also integrates with the Ensembl BioMart API to convert Ensembl gene IDs to human-readable gene names.

Features
- Feature selection using ANOVA (`SelectKBest`)
- Class imbalance handled via `SMOTE`
- Classification using Random Forest and SVM
- Evaluation with classification reports and ROC-AUC
- Feature importance ranking
- Gene name mapping via BioMart**
- Leave-One-Out Cross Validation (LOOCV) and multiple iterations for robustness
- Visualization: ROC curves and feature importance
- Cross-platform compatible setup

---

    📁 Project Structure

```
ML-into-Transcriptomics/ 
├── models/            
├── MultipleModels     
├── RFSVM            
├── requirements.txt  
└── mkdir.sh          
```

---

## ⚙️ Setup Instructions

### ✅ Option 1: Conda (Recommended for macOS/Linux/Windows)

```bash
conda env create -f environment.yml
conda activate transcriptomics
```

### ✅ Option 2: pip + virtualenv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> 🍎 **macOS only:**  
If using pip, install `libomp` for XGBoost:
```bash
brew install libomp
```

---

## 🧠 How to Run

Make sure your dataset (like `Rsem_count.csv`) path is correct.

Then run:

```bash
python scripts/bio-markers.py
```

You’ll get:
- Classification performance from RF and SVM
- ROC plots
- Top 10 features per model
- Human-readable gene names from Ensembl
- Common top biomarkers between models

---

## 📊 Output Example

- 🎯 Model Accuracy
- 🧬 Top Biomarkers:
  ```
  1. TP53 (ENSG00000141510)
  2. BRCA1 (ENSG00000012048)
  ...
  ```

- 📈 ROC Curve saved to `results/`

---

## 🤝 Contributions

Feel free to open issues or submit pull requests to:
- Add more ML models (e.g., XGBoost, LightGBM)
- Integrate DESeq2 outputs
- Enhance visualization

---

## 🧬 Credits

Built by Saiesh Prabhu, Siddique, Md Abu T   
Powered by: scikit-learn, imbalanced-learn, biomart, matplotlib, pandas, numpy
