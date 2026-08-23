# Final Audit

Audit evidence is based on the current workspace and the commands recorded in the project history. `BLOCKED` is used only for the missing exact generator block referenced by the supplied brief.

| Criterion | Evidence | Command | Result | Status |
|---|---|---|---|---|
| R01 deterministic 6000-row dataset | `orders_dataset.csv`, `generate_orders.py` | `python generate_orders.py` | 6000 rows | PASS |
| R02 exact unseen generator constants/logic | Supplied attachment | Source comparison | Exact source block absent | BLOCKED |
| R03 13 columns | `orders_dataset.csv` | Pandas shape assertion | (6000, 13) | PASS |
| R04 dataset EDA metrics | `reports/return_risk_report.json` | Part 1 evaluation | Actual rates saved | PASS |
| R05 MAR missingness | `reports/return_risk_analysis.json` | Part 1 analysis | MAR; gap 0.183144 | PASS |
| R06 stratified split | `train_return_risk.py` | Part 1 training | 80/20, seed 42 | PASS |
| R07 leakage-safe preprocessing | `part1_return_risk/utils.py` | `pytest -q` | Pipeline contract passes | PASS |
| R08 Dummy baseline | `return_risk_report.json` | Part 1 training | Accuracy 0.78083; F1 0 | PASS |
| R09 Logistic threshold sweep | `return_risk_report.json` | Part 1 training | Best threshold 0.49 | PASS |
| R10 RF grid/CV/test | `return_risk_report.json` | Part 1 training | CV 0.62552; test 0.63665 | PASS |
| R11 importance comparison | `return_risk_analysis.json` | Part 1 analysis | Impurity/permutation saved | PASS |
| R12 subgroup analysis | `return_risk_analysis.json` | Part 1 analysis | Electronics recall 0.30952 | PASS |
| R13 saved RF pipeline | `models/return_risk_model.pkl` | `pytest -q` | Loads successfully | PASS |
| R14 RF threshold metadata | `models/return_risk_metadata.json` | `pytest -q` | t*=0.53 | PASS |
| R15 Fashion-MNIST split | `part2_product_classifier/train_classifier.py` | `verify_splits.py` | 55000/5000/10000 | PASS |
| R16 image transforms | `train_classifier.py`, `predict.py` | Smoke test | 3 channels, 224, ImageNet normalization | PASS |
| R17 transfer learning/cache | `train_classifier.py`, cache artifacts | Colab training | CUDA; Stage A 0.9174 | PASS |
| R18 conditional fine-tune | training report | Colab training | Skipped at 0.9174 | PASS |
| R19 untouched test evaluation | `product_classifier_evaluation.json` | `evaluate.py` | Accuracy 0.9083; 10x10 matrix | PASS |
| R20 reusable image model API | `predict.py`, checkpoint | `pytest -q` | Artifact and API validated | PASS |
| R21 real sample PNGs | `data/sample_images/` | `export_samples.py` | 10 images | PASS |
| R22 12 policy documents | `knowledge_base/policies.json` | Agent smoke test | 12 documents | PASS |
| R23 MiniLM/FAISS retrieval | `rag.py`, FAISS index | `python -m part3_support_agent.rag` | Index built and loaded | PASS |
| R24 dynamic risk tool | `tools.py` | `pytest -q` | Real probability and threshold | PASS |
| R25 image tool | `tools.py`, sample PNG | `run_transcripts.py` | Real classification | PASS |
| R26 conditional LangGraph | `graph.py` | Agent smoke test | Conditional graph compiled | PASS |
| R27 conversational state | `state.py`, transcripts | `pytest -q` | Same-thread carry/fresh reset | PASS |
| R28 4S and few-shot prompts | `prompts.py` | Source inspection | Role, 4S, examples present | PASS |
| R29 schema/MOCK_LLM | `schemas.py`, `mock_llm.py` | `pytest -q` | Structured deterministic output | PASS |
| R30 input guardrail | `guardrails.py` | `test_guardrails.py` | Injection blocked | PASS |
| R31 groundedness refusal | transcript 06 | `run_transcripts.py` | Refusal generated | PASS |
| R32 eight transcripts | `transcripts/` | `run_transcripts.py` | 9 generated | PASS |
| R33 retrieval metrics | `knowledge_base/retrieval_eval.json` | `eval_retrieval.py` | P@3 0.400; R@3 1.000 | PASS |
| R34 automated tests | `tests/` | `pytest -q` | 9 passed | PASS |
| R35 reproducibility/config/logging/errors/README | README and modules | Setup/run commands | Documented | PASS |
| R36 final audit | This document | Audit review | All criteria recorded | PASS |
| R37 Git workflow | Git history | `git log --graph --all` | Feature commits and merge history | PASS |

## Residual Risk

The exact generator block is unavailable in the supplied specification, so R02 cannot be independently verified. The FAISS index is intentionally rebuilt locally from the committed knowledge base rather than storing a generated binary.