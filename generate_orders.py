"""Generate the deterministic order dataset required by the capstone brief."""

from pathlib import Path

import numpy as np
import pandas as pd

N = 6000
SEED = 42
CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
CATEGORY_PROBABILITIES = [0.32, 0.22, 0.18, 0.18, 0.10]
PAYMENT_METHODS = ["COD", "Prepaid_Card", "Prepaid_UPI", "Wallet"]
PAYMENT_PROBABILITIES = [0.42, 0.24, 0.24, 0.10]
BASE_PRICE = {
    "Apparel": (400, 2200), "Electronics": (1200, 45000),
    "Home": (300, 8000), "Footwear": (500, 4500), "Beauty": (150, 2500),
}


def generate_orders(output_path: Path = Path("orders_dataset.csv")) -> pd.DataFrame:
    """Generate and save a deterministic 6,000-row, 13-column order table."""
    rng = np.random.default_rng(42)
    category = rng.choice(CATEGORIES, size=N, p=CATEGORY_PROBABILITIES)
    payment_method = rng.choice(PAYMENT_METHODS, size=N, p=PAYMENT_PROBABILITIES)
    price_inr = np.round(np.array([rng.uniform(*BASE_PRICE[c]) for c in category]), 0)
    discount_pct = np.clip(rng.normal(22, 15, N), 0, 75)
    customer_tenure_days = np.clip(rng.exponential(380, N), 1, 2500).round(0)
    num_previous_orders = np.clip(
        (customer_tenure_days / 45) + rng.normal(0, 2, N), 0, None
    ).round(0)
    base_return_rate = np.clip(rng.beta(1.5, 9, N), 0, 1)
    num_previous_returns = np.round(
        base_return_rate * num_previous_orders
    ).clip(0, num_previous_orders)
    delivery_distance_km = np.clip(rng.gamma(3, 90, N), 2, 2200).round(1)
    delivery_days = np.clip(rng.normal(4.5, 2.2, N), 1, 21).round(0)
    is_weekend_order = rng.integers(0, 2, N)
    customer_rating = rng.integers(1, 6, N).astype(float)
    missing_mask = rng.random(N) < np.where(payment_method == "COD", 0.22, 0.06)
    customer_rating[missing_mask] = np.nan

    fit_risk_category = np.isin(category, ["Apparel", "Footwear"]).astype(float)
    previous_return_ratio = np.where(
        num_previous_orders > 0,
        num_previous_returns / np.maximum(num_previous_orders, 1),
        0,
    )
    z = (
        -2.2 + 1.9 * previous_return_ratio + 0.55 * fit_risk_category
        + 0.014 * (discount_pct - 20) / 10
        + 0.9 * (payment_method == "COD").astype(float)
        + 0.10 * (delivery_days - 4.5) / 2
        + 0.30 * (price_inr / BASE_PRICE["Electronics"][1])
        + 0.05 * is_weekend_order
        - 0.15 * np.tanh(customer_tenure_days / 500)
    )
    return_probability = 1 / (1 + np.exp(-z))
    returned = (rng.random(N) < return_probability).astype(int)

    data = pd.DataFrame(
        {
            "order_id": np.arange(1, N + 1),
            "product_category": category,
            "price_inr": price_inr,
            "discount_pct": discount_pct,
            "payment_method": payment_method,
            "customer_tenure_days": customer_tenure_days.astype(int),
            "num_previous_orders": num_previous_orders.astype(int),
            "num_previous_returns": num_previous_returns,
            "delivery_distance_km": delivery_distance_km,
            "delivery_days": delivery_days,
            "is_weekend_order": is_weekend_order,
            "customer_rating": customer_rating,
            "returned": returned,
        }
    )
    data.to_csv(output_path, index=False)
    return data


if __name__ == "__main__":
    orders = generate_orders()
    print(f"rows={len(orders)} columns={len(orders.columns)}")
    print(f"return_rate={orders['returned'].mean():.4f}")
    print(f"rating_missing_rate={orders['customer_rating'].isna().mean():.4f}")
    print("missing_rate_by_payment=")
    print(orders.groupby("payment_method")["customer_rating"].apply(lambda s: s.isna().mean()))
