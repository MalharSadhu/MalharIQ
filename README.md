# MalharIQ: Autonomous Enterprise MLOps & Analytics Advisory Platform

MalharIQ is an enterprise-grade MLOps platform engineered to automate multi-modal asset ingestion, statistical validation, algorithm benchmarking across strict 70/15/15 partitions, quantitative regulatory governance (EEOC 80% rule), and client-ready deliverable synthesis.

---

## Quickstart Guide (Local Execution)

Follow these exact steps to run MalharIQ locally on any machine (Windows, macOS, Linux):

Step 1. Clone the Repository:
git clone https://github.com/MalharSadhu/MalharIQ.git
cd MalharIQ

Step 2. Set Up a Virtual Environment:
Windows (PowerShell):
python -m venv env
.\env\Scripts\activate

macOS / Linux:
python3 -m venv env
source env/bin/activate

Step 3. Install Dependencies:
pip install --upgrade pip
pip install -r requirements.txt

Step 4. (Optional) Run Local LLM for Copilot & AI Synthesis:
MalharIQ uses a lightweight local LLM (llama3.2:1b) for unstructured document parsing and the advisory copilot.
- Download and install Ollama: https://ollama.com
- Pull and start the model in your terminal:
ollama run llama3.2:1b
(Note: If Ollama is not installed or running, MalharIQ automatically defaults to robust deterministic rule-based synthesis. The app will never crash.)

Step 5. Launch the Dashboard:
streamlit run app.py
The application will open automatically in your browser at http://localhost:8501.

---

## Step-by-Step Platform Execution Walkthrough

Once the dashboard is running in your browser:

Phase 1. Ingest an Enterprise Asset:
- Structured Path: Under the Structured Tabular Ingestion (CSV / Excel) tab, upload a dataset (.csv, .xlsx, or .xls).
- Unstructured Path: Under the Unstructured Asset Parsing tab, upload raw text (.txt), logs (.json, .log), documents (.pdf), or images (.png, .jpg), and click Synthesize Dataset.

Phase 2. Configure Analytical Directive:
- Review the raw data profile (total records, dimensions, missing values).
- Select a pipeline mode:
  * Autonomous Intent Discovery (Zero Configuration): The engine inspects target properties, cardinality, and distribution to infer whether the task is Regression, Classification, or Clustering automatically.
  * Custom Strategic Directive: Specify an explicit business goal (e.g., Predict customer churn risk or Segment vehicles by efficiency).

Phase 3. Run the Autonomous Advisory Pipeline:
- Click Execute Advisory Pipeline.
- The status tracker will sequentially display progress through:
  1. Statistical profiling and metadata inspection.
  2. Objective and target inference.
  3. Preprocessing, anomaly resolution, and sparse data synthesis.
  4. 70% Train, 15% Validation, and 15% Holdout Test partitioning.
  5. Statistical Quality Gates (KMO, Bartlett test, Factor loadings).
  6. Multi-architecture algorithm tournaments and validation benchmarking.
  7. Client-ready document synthesis.

Phase 4. Explore Advisory Workspaces & Export Deliverables:
- Executive Memo: Review high-level verdicts, strategic action items, and download the synthesized Executive Memo (PDF) or Client Presentation (.pptx).
- Engineering Deep Dive: Review technical architecture logs, preprocessing audit trails, and production monitoring plans.
- Algorithm Tournament: Compare candidate models ranked on the 15% validation split and verified on the 15% holdout test partition.
- Quality Gates: Inspect sampling adequacy metrics and orthogonal factor loading matrices.
- Visual Analytics: View empirical diagnostic plots:
  * Regression: Actual vs. Predicted scatter plot with ideal fit line, Residual error distribution, and Predictive driver impacts.
  * Classification: Feature impact charts and validation accuracy/F1 breakdowns.
  * Clustering: Multi-feature segment separation plots.
- Pattern Mining: Examine discovered cross-variable association rules.
- Governance & Bias Audit: Select sensitive attributes to evaluate demographic parity under the EEOC 80% Four-Fifths rule.
- Ask MalharIQ Copilot: Use the local conversational assistant to query run metadata, partition splits, and governance findings.

---

## In-Depth Platform Architecture

MalharIQ implements an end-to-end MLOps lifecycle divided into seven modular subsystems:

[ Multi-Modal Ingestion ] -> [ Autonomous Cleaner & Synthesizer ] -> [ 70/15/15 Partitioner ]
                                                                             |
[ Client Deliverables ]   <-   [ Algorithm Tournament ]   <-   [ Statistical Quality Gates ]
  (PDF / PPTX / Copilot)          (Validation Ranking)               (KMO / Bartlett / PCA)

1. Multi-Modal Ingestion Engine:
Standardizes structured tabular sheets (.csv, .xlsx, .xls) and converts unstructured text, PDFs, logs, and image channel statistics into structured tabular schemas via local LLM parsing (llama3.2:1b) with zero cloud data leakage.

2. Autonomous Sparse Data Synthesizer:
Micro-datasets (< 30 samples) crash standard train/validation/test holdout splits. MalharIQ detects sparse datasets and bootstraps statistically consistent records up to 100 samples, preserving numerical covariance, Gaussian variance, and categorical distribution frequencies.

3. Strict 70 / 15 / 15 Partitioning Architecture:
Eliminates data leakage by strictly isolating:
- 70% Training Set: Used exclusively to fit candidate models.
- 15% Validation Set: Used for hyperparameter tuning and leaderboard tournament ranking.
- 15% Test Set: An untouched holdout partition used solely for final empirical reporting.

4. Multi-Algorithm Tournaments:
Benchmarks diverse model families in parallel:
- Regression: Random Forest, Gradient Boosting, Ridge, Lasso, OLS Linear Regression, and Multi-Layer Perceptron (ANN).
- Classification: Random Forest, XGBoost, Logistic Regression, Support Vector Machines (SVM), KNN, Deep MLP, Soft Voting Ensembles, and Stacking Classifiers.
- Clustering: K-Means across variable cluster counts and Hierarchical Agglomerative Clustering (Ward, Complete, Average).

5. Statistical Quality Gates:
- Kaiser-Meyer-Olkin (KMO) Measure of Sampling Adequacy: Validates whether variables share sufficient common variance for clustering.
- Bartlett Test of Sphericity: Confirms that correlation matrices are statistically non-orthogonal (p < 0.05).
- Orthogonal Factor Analysis: Isolates underlying latent drivers and reduces dimensionality.

6. Quantitative Regulatory Governance:
Evaluates adverse impact across sensitive demographic and cohort proxies using the EEOC 80% Four-Fifths Rule. Flags selection parity alerts and dynamically computes disparity ratios against the regulatory 0.80 benchmark floor.

7. Automated Deliverable Synthesis:
Programmatically compiles and embeds empirical diagnostic charts into downloadable executive reports:
- Formatted multi-section PDF Briefs (fpdf2).
- Styled executive slide decks (python-pptx).

---

## Key Platform Advantages

1. Eliminates Algorithmic Silos: Bridges the gap between raw data and executive decision-making by replacing manual notebook experimentation with an end-to-end automated pipeline.
2. Guaranteed Mathematical Integrity: Prevents data leakage through strict 70/15/15 train-validation-test holdout partitioning.
3. Multi-Modal Flexibility: Unifies structured CSV/Excel parsing and unstructured text/PDF/image ingestion into a single workflow.
4. Built-In Regulatory Protection: Embeds quantitative algorithmic governance (EEOC 80% rule) directly into the model promotion lifecycle.
5. Zero-Configuration Execution: Autonomous intent discovery infers ML tasks, target features, and cleaning recipes without manual configuration.
6. Instant Executive Communication: Generates board-ready PowerPoint decks and PDF memos with embedded diagnostic plots in seconds, eliminating manual reporting overhead.
7. Local & Air-Gapped Operation: Runs 100% locally via Ollama without sending enterprise data to external third-party cloud APIs.

---

## Repository File Structure

MalharIQ/
|-- app.py                  # Main Streamlit enterprise advisory application
|-- ml_engine.py            # Model tournaments, 70/15/15 partitions, and ML logic
|-- core_cleaner.py         # Autonomous intent inference and data cleaning recipes
|-- document_exporter.py    # Automated PDF brief and PPTX presentation generators
|-- governance_audit.py     # EEOC 80% disparate impact and regulatory bias audit
|-- reporter.py             # Dual executive and technical report synthesizer
|-- stats_validator.py      # Statistical quality gates (KMO, Bartlett, Factor Analysis)
|-- requirements.txt        # Cross-platform Python dependencies
`-- README.md               # Setup and platform documentation
