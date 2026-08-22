"""Generate the provisional deterministic order dataset for project initialization.

The capstone brief requires an exact generator block that was not included in the
supplied attachment. This implementation preserves the explicit N and RNG seed,
and keeps all assumptions visible until the reference block is supplied.
"""

from pathlib import Path

import numpy as np
import pandas as pd

N = 6000
SEED = 42
CATEGORIES = [
    "Electronics",
    "Fashion",
    "Home",
    "Beauty",
    "Books",
    "Grocery",
    "Footwear",
]
CATEGORY_PROBABILITIES = [0.20, 0.20, 0.15, 0.10, 0.10, 0.10, 0.15]
PAYMENT_METHODS = ["COD", "Credit Card", "Debit Card", "UPI", "Wallet"]
PAYMENT_PROBABILITIES = [0.35, 0.20, 0.15, 0.25, 0.05]


def generate_orders(output_path: Path = Path("orders_dataset.csv")) -> pd.DataFrame:
    """Generate and save a deterministic 6,000-row, 13-column order table."""
    rng = np.random.default_rng(42)
    category = rng.choice(CATEGORIES, size=N, p=CATEGORY_PROBABILITIES)
    payment_method = rng.choice(
        PAYMENT_METHODS, size=N, p=PAYMENT_PROBABILITIES
    )
    price_inr = np.round(rng.lognormal(mean=7.2, sigma=0.8, size=N), 2)
    discount_pct = np.round(rng.uniform(0, 70, size=N), 2)
    customer_tenure_days = rng.integers(30, 2500, size=N)
    num_previous_returns = rng.poisson(0.8, size=N)
    delivery_days = rng.integers(1, 15, size=N)
    quantity = rng.integers(1, 5, size=N)
    customer_age = rng.integers(18, 70, size=N)
    order_value_inr = np.round(price_inr * quantity * (1 - discount_pct / 100), 2)
    is_cod = (payment_method == "COD").astype(int)

    logit = (
        -2.0
        + 0.75 * is_cod
        + 0.45 * (discount_pct > 45)
        + 0.55 * (num_previous_returns >= 2)
        + 0.30 * (delivery_days >= 10)
        + 0.20 * (category == "Fashion")
    )
    return_probability = 1 / (1 + np.exp(-logit))
    returned = rng.binomial(1, return_probability)

    # Missingness depends on observed payment method, which is MAR.
    missing_probability = np.where(payment_method == "COD", 0.28, 0.08)
    customer_rating = np.round(rng.uniform(1, 5, size=N), 1)
    customer_rating[rng.random(N) < missing_probability] = np.nan

    data = pd.DataFrame(
        {
            "order_id": np.arange(100000, 100000 + N),
            "product_category": category,
            "payment_method": payment_method,
            "price_inr": price_inr,
            "discount_pct": discount_pct,
            "customer_tenure_days": customer_tenure_days,
            "num_previous_returns": num_previous_returns,
            "delivery_days": delivery_days,
            "quantity": quantity,
            "customer_age": customer_age,
            "order_value_inr": order_value_inr,
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
