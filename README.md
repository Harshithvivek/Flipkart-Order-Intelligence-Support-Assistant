# Flipkart Order Intelligence & Support Assistant

A reproducible three-part capstone combining return-risk machine learning, Fashion-MNIST transfer learning, and a grounded LangGraph support agent.

## Project Status

Initialization is in progress. See [docs/REQUIREMENTS_MATRIX.md](docs/REQUIREMENTS_MATRIX.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The exact order-generator block referenced by the supplied brief was not present in the attachment, so the initial dataset generator is explicitly provisional until that source is provided.

## Initial Command

```powershell
python generate_orders.py
```

The command creates `orders_dataset.csv` and prints deterministic shape and missingness checks. Later stages will add model artifacts, retrieval indexes, reports, transcripts, and tests.
