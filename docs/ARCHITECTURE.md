# Flipkart Order Intelligence & Support Assistant

## System Flow

```text
Part 1: deterministic orders -> leakage-safe ML pipeline -> models/return_risk_model.pkl
                                                    |
                                                    v
Part 3: LangGraph support agent -> return-risk tool
                         |
Part 2: Fashion-MNIST ResNet-18 -> models/product_classifier.pt
                         |
                         +----> product image tool

Part 3 policy requests -> MiniLM embeddings -> FAISS document index -> grounded response
```

## Components

- **Part 1** generates and analyzes order-return data, compares baselines, tunes logistic and random-forest thresholds, evaluates subgroups, and persists one fitted scikit-learn pipeline plus RF threshold metadata.
- **Part 2** uses Fashion-MNIST and a pretrained ResNet-18. A frozen-backbone feature cache is used for CPU-friendly training, with late-layer fine-tuning conditional on validation performance.
- **Part 3** routes policy, return-risk, and product-image intents through conditional LangGraph edges. Policy responses use parent-document-aware FAISS retrieval; tools call the saved Part 1 and Part 2 artifacts.
- **Reproducibility** is controlled by explicit seeds, relative paths, environment configuration, generated reports, and executable tests.

## Current Boundary

The attachment does not include the referenced exact dataset-generator specification. `generate_orders.py` uses a documented provisional schema and deterministic logic solely to establish the repository and verify the requested 6,000-row/13-column shape. It must be reconciled with the source generator before Part 1 is considered complete.
