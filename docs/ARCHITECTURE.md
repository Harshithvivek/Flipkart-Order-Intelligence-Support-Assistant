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

## Reproducibility Boundary

`generate_orders.py` contains the recovered reference generator: five categories, four payment methods, `N=6000`, `np.random.default_rng(42)`, observed-payment-method MAR missingness, and the specified return equation.
