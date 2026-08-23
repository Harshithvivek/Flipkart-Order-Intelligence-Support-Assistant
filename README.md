# Flipkart Order Intelligence & Support Assistant

A reproducible three-part capstone combining return-risk machine learning, Fashion-MNIST transfer learning, and a grounded LangGraph support agent.

## Project Status

Part 1 is implemented and Part 2 training has completed on CUDA. Stage A reached 91.74% validation accuracy on the required 55,000/5,000 split. The first uploaded test report was invalid because the checkpoint was assembled with a random backbone; that root cause is fixed in `part2_product_classifier/train_classifier.py`, and Part 2 must be rerun before its test metrics are accepted.

See [docs/REQUIREMENTS_MATRIX.md](docs/REQUIREMENTS_MATRIX.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The exact order-generator block referenced by the brief was not included in the supplied attachment, so the deterministic Part 1 generator is explicitly provisional.

## Initial Command

```powershell
python generate_orders.py
```

The command creates `orders_dataset.csv` and prints deterministic shape and missingness checks.

## Part 2 Rerun

In Colab with a CUDA runtime, pull the checkpoint fix and rerun the model and untouched test evaluation:

```python
!git pull origin main
!python -m part2_product_classifier.train_classifier
!python -m part2_product_classifier.evaluate
!python -m part2_product_classifier.export_samples
```
