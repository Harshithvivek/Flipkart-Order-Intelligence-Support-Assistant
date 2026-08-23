# Requirements Matrix

This matrix is derived from the supplied capstone brief. `BLOCKED` means the brief refers to material that was not included in the attachment, or an artifact/result that can only be verified after later execution.

| ID | Requirement | Part | Implementation File | Verification Method | Status |
|---|---|---|---|---|---|
| R01 | Deterministic order dataset with `N=6000` and `np.random.default_rng(42)` | 1 | `generate_orders.py` | Run generator; inspect shape and seed behavior | PASS |
| R02 | Use the exact specified category list, category probabilities, payment methods, payment probabilities, and generation logic | 1 | `generate_orders.py` | Compare constants and generated output with full source specification | BLOCKED: exact generator block absent from supplied attachment |
| R03 | Dataset has exactly 13 columns and saved as `orders_dataset.csv` | 1 | `generate_orders.py` | Pandas shape check | PASS |
| R04 | Report return rate, rating missingness, category return rates, payment return rates | 1 | `part1_return_risk/evaluate_return_risk.py` | Execute evaluation report | NOT STARTED |
| R05 | Classify missingness as MAR using observed payment method and measure COD/non-COD gap | 1 | `part1_return_risk/analyze_return_risk.py` | Execute missingness analysis | NOT STARTED |
| R06 | 80/20 stratified split with `random_state=42` | 1 | `part1_return_risk/train_return_risk.py` | Inspect training run | NOT STARTED |
| R07 | Leakage-safe `ColumnTransformer` with median numeric imputation, scaling, categorical most-frequent imputation, and one-hot encoding | 1 | `part1_return_risk/utils.py` | Pipeline inspection and test | NOT STARTED |
| R08 | Dummy most-frequent baseline with accuracy and returned-class F1; discuss imbalance and zero recall | 1 | `part1_return_risk/train_return_risk.py` | Training report | NOT STARTED |
| R09 | Balanced logistic regression, threshold-0.5 metrics, and 0.10-0.90 sweep at <=0.02 steps | 1 | `part1_return_risk/train_return_risk.py` | Training report | NOT STARTED |
| R10 | Balanced random forest with GridSearchCV, required grid, ROC-AUC scoring, and 5-fold StratifiedKFold | 1 | `part1_return_risk/train_return_risk.py` | CV/test report | NOT STARTED |
| R11 | Compare impurity and held-out permutation importance; report top five and required features | 1 | `part1_return_risk/analyze_return_risk.py` | Importance report | NOT STARTED |
| R12 | Subgroup precision/recall by category and payment method; identify weak subgroup and concrete remedy | 1 | `part1_return_risk/analyze_return_risk.py` | Subgroup report | NOT STARTED |
| R13 | Save final fitted tuned RF pipeline to `models/return_risk_model.pkl` | 1 | `part1_return_risk/train_return_risk.py` | Load artifact test | NOT STARTED |
| R14 | Sweep RF probabilities and save actual `t*_rf` to machine-readable metadata | 1 | `part1_return_risk/train_return_risk.py` | Metadata inspection | NOT STARTED |
| R15 | Fashion-MNIST 60,000/10,000 split with stratified validation >=5,000 | 2 | `part2_product_classifier/train_classifier.py`, `verify_splits.py` | Dataset/split report | IN PROGRESS |
| R16 | Grayscale-to-three-channel resize and ImageNet normalization | 2 | `part2_product_classifier/train_classifier.py`, `predict.py` | Transform inspection | IN PROGRESS |
| R17 | Pretrained ResNet-18 transfer learning with frozen backbone, Adam, documented configuration, and feature cache | 2 | `part2_product_classifier/train_classifier.py` | Training logs/artifacts | IN PROGRESS |
| R18 | Fine-tune late layers only if validation accuracy is below 80%; report before/after actual values | 2 | `part2_product_classifier/train_classifier.py` | Training report | IN PROGRESS |
| R19 | Untouched test evaluation with accuracy, 10x10 confusion matrix, per-class metrics, and actual confusion pairs | 2 | `part2_product_classifier/evaluate.py` | Evaluation report | NOT STARTED: rebuild pending execution |
| R20 | Save reusable classifier to `models/product_classifier.pt` with `load_product_classifier()` and `predict_product_image()` | 2 | `part2_product_classifier/predict.py` | Model loading/prediction tests | IN PROGRESS |
| R21 | Export >=5 real Fashion-MNIST test images under `data/sample_images/` | 2 | `part2_product_classifier/export_samples.py` | File provenance check | NOT STARTED: rebuild pending execution |
| R22 | At least 12 policy documents, required coverage, and retained document/chunk IDs | 3 | `knowledge_base/policies.json` | KB validation | NOT STARTED |
| R23 | Local MiniLM embeddings and FAISS index with reproducible build | 3 | `part3_support_agent/rag.py` | Index build/retrieval tests | NOT STARTED |
| R24 | Dynamic return-risk tool loads fitted pipeline and RF threshold; no hardcoded predictions | 3 | `part3_support_agent/tools.py` | Tool tests | NOT STARTED |
| R25 | Image tool calls the Part 2 prediction function and reads real PNG input | 3 | `part3_support_agent/tools.py` | Tool tests | NOT STARTED |
| R26 | LangGraph conditional intent routing across policy, return-risk, and product paths | 3 | `part3_support_agent/graph.py` | Routing tests | NOT STARTED |
| R27 | Short-term conversational state and fresh-conversation reset behavior | 3 | `part3_support_agent/state.py` | Multi-turn tests/transcripts | NOT STARTED |
| R28 | 4S prompt annotation, role prompt, and two few-shot intent examples | 3 | `part3_support_agent/prompts.py` | Prompt inspection/transcripts | NOT STARTED |
| R29 | Strict final schema with allowed source values and deterministic MOCK_LLM default | 3 | `part3_support_agent/schemas.py`, `mock_llm.py` | Schema/agent tests | NOT STARTED |
| R30 | Input injection guardrail for specified patterns | 3 | `part3_support_agent/guardrails.py` | Guardrail tests/transcript | NOT STARTED |
| R31 | Similarity-threshold groundedness refusal for unsupported policy questions | 3 | `part3_support_agent/agent.py` | Refusal test/transcript | NOT STARTED |
| R32 | At least 8 transcripts generated from actual execution | 3 | `transcripts/` | Transcript generation command | NOT STARTED |
| R33 | At least 5 document-level Precision@3 and Recall@3 evaluations with arithmetic and averages | 3 | `knowledge_base/retrieval_eval.json` | Retrieval evaluation command | NOT STARTED |
| R34 | Automated tests for artifacts, tools, routing, guardrails, state, schema, and retrieval | All | `tests/` | `pytest` | NOT STARTED |
| R35 | Reproducible setup, configuration, logging, errors, and README | All | `README.md`, `.env.example`, package modules | Clean-environment run | NOT STARTED |
| R36 | Final audit with PASS/FAIL/BLOCKED evidence for every criterion | All | `docs/FINAL_AUDIT.md` | Audit review | NOT STARTED |
| R37 | Feature branch, >=2 meaningful commits, merge into main, and visible graph | All | Git history | `git log --graph --all` | NOT STARTED |

## Initialization Note

The supplied attachment references an earlier full specification but does not contain the promised exact order-generator code/constants. R02 therefore remains blocked until that source block is supplied. The initial generator is intentionally marked provisional; no downstream model claim should treat it as the required final dataset until R02 is resolved.
