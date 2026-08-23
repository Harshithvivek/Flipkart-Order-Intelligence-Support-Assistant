# Flipkart Order Intelligence & Support Assistant

A reproducible three-part capstone combining return-risk machine learning, Fashion-MNIST transfer learning, and a grounded LangGraph support agent.

## Project Status

Part 1 is implemented and Part 2 is complete. The corrected Part 2 run used CUDA, reached 91.74% Stage A validation accuracy, and achieved 90.83% accuracy on the untouched 10,000-image test set. The support-agent foundation and actual FAISS retrieval/transcript checks are also complete.

See [docs/REQUIREMENTS_MATRIX.md](docs/REQUIREMENTS_MATRIX.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The deterministic Part 1 generator matches the recovered reference specification.

## Initial Command

```powershell
python generate_orders.py
```

The command creates `orders_dataset.csv` and prints deterministic shape and missingness checks.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The default agent mode is `MOCK_LLM`; it requires no API key or paid service. Build the local FAISS index before policy queries. The first retrieval run downloads `all-MiniLM-L6-v2` when it is not cached.

## Part 1 Results

The generated dataset contains 6,000 rows and 13 columns. The observed return rate is 0.2275, rating missingness is MAR, and the COD versus non-COD missingness gap is 16.77 percentage points. The DummyClassifier has accuracy 0.78083 and returned-class F1 0.0. Random Forest CV ROC-AUC is 0.61797, held-out ROC-AUC is 0.61755, and the RF-specific threshold `t*_rf` is 0.47.

## Verified Results

- Part 2: CUDA, 55,000 train / 5,000 validation / 10,000 test, validation accuracy 0.9174, test accuracy 0.9083.
- Part 3 retrieval: all five evaluation queries achieved Recall@3 of 1.000; average Precision@3 was 0.400.
- Part 3 transcripts: nine real MOCK_LLM scenarios generated, including policy refusal, prompt injection, image classification, and same-thread versus fresh-thread state.

## Part 3 Commands

```powershell
python -m part3_support_agent.rag
python -m part3_support_agent.eval_retrieval
python -m part3_support_agent.run_transcripts
```

## Testing

```powershell
pytest -q
```

The current suite covers dataset/model artifacts, dynamic tools, conditional routing, schema validation, injection blocking, groundedness refusal, state persistence/reset, and retrieval evaluation.

## Part 2 Rerun

In Colab with a CUDA runtime, pull the checkpoint fix and rerun the model and untouched test evaluation:

```python
!git pull origin main
!python -m part2_product_classifier.train_classifier
!python -m part2_product_classifier.evaluate
!python -m part2_product_classifier.export_samples
```
