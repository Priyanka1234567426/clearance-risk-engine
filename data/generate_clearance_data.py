"""
Synthetic Clearance Dataset Generator
Creates ~10,000 shipment records with realistic distributions and signal.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

commodity_categories = ['Apparel', 'Electronics', 'Pharma', 'Industrial', 'Food']
commodity_probs = [0.35, 0.25, 0.10, 0.20, 0.10]

consignor_countries = ['IN', 'CN', 'VN', 'BD', 'TH']
consignor_probs = [0.45, 0.30, 0.10, 0.10, 0.05]

consignee_countries = ['US', 'CA', 'UK', 'DE']
consignee_probs = [0.70, 0.10, 0.10, 0.10]

hs_prefix_map = {
    'Apparel': ['6109', '6110', '6204', '6203'],
    'Electronics': ['8517', '8528', '8471', '8473'],
    'Pharma': ['3004', '3003', '3002'],
    'Industrial': ['8479', '7308', '7326'],
    'Food': ['2106', '0901', '1905']
}

data = {
    'shipment_id': [f'SHIP{i:06d}' for i in range(N)],
    'consignor_country': np.random.choice(consignor_countries, N, p=consignor_probs),
    'consignee_country': np.random.choice(consignee_countries, N, p=consignee_probs),
    'commodity_category': np.random.choice(commodity_categories, N, p=commodity_probs),
    'declared_value_usd': np.round(np.random.lognormal(mean=6.5, sigma=1.2, size=N), 2),
    'weight_kg': np.round(np.random.lognormal(mean=2.0, sigma=1.0, size=N), 2),
    'num_line_items': np.random.choice([1, 2, 3, 4, 5, 6, 8, 10, 15], N,
                                        p=[0.30, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]),
    'consignor_history_score': np.clip(np.round(np.random.normal(75, 18, N)), 0, 100),
    'is_first_time_consignee': np.random.choice([0, 1], N, p=[0.70, 0.30]),
    'documentation_completeness_score': np.clip(np.round(np.random.normal(82, 15, N)), 0, 100),
}

df = pd.DataFrame(data)
df['lane'] = df['consignor_country'] + '-' + df['consignee_country']
df['hs_code'] = df['commodity_category'].apply(
    lambda c: np.random.choice(hs_prefix_map[c]) + str(np.random.randint(10, 99))
)

# Target: was_held_at_customs
hold_prob = (
    0.05
    + 0.20 * (df['documentation_completeness_score'] < 70).astype(int)
    + 0.15 * (df['consignor_history_score'] < 60).astype(int)
    + 0.10 * df['is_first_time_consignee']
    + 0.12 * (df['commodity_category'] == 'Pharma').astype(int)
    + 0.08 * (df['commodity_category'] == 'Food').astype(int)
    + 0.10 * (df['declared_value_usd'] > 5000).astype(int)
    + 0.05 * (df['num_line_items'] > 5).astype(int)
)
hold_prob = np.clip(hold_prob, 0, 0.85)
df['was_held_at_customs'] = (np.random.random(N) < hold_prob).astype(int)

# Target: hold_reason
def assign_hold_reason(row):
    if row['was_held_at_customs'] == 0:
        return 'None'
    weights = {
        'Missing Docs': 1.0 + 3 * (row['documentation_completeness_score'] < 70),
        'HS Mismatch': 1.0 + 2 * (row['num_line_items'] > 5),
        'Value Discrepancy': 1.0 + 2 * (row['declared_value_usd'] > 5000),
        'Restricted Goods': 1.0 + 4 * (row['commodity_category'] in ['Pharma', 'Food']),
    }
    reasons = list(weights.keys())
    probs = np.array(list(weights.values()))
    probs = probs / probs.sum()
    return np.random.choice(reasons, p=probs)

df['hold_reason'] = df.apply(assign_hold_reason, axis=1)

# Target: clearance_time_hours
clearance_time = (
    8
    + 30 * df['was_held_at_customs']
    + 0.3 * (100 - df['documentation_completeness_score'])
    + 5 * df['is_first_time_consignee']
    + 0.5 * df['num_line_items']
    + np.random.normal(0, 5, N)
)
df['clearance_time_hours'] = np.round(np.clip(clearance_time, 1, 200), 1)

df.to_csv('data/clearance_shipments.csv', index=False)
print(f"Generated {len(df)} shipments.")
print(f"\nClass balance (was_held_at_customs):")
print(df['was_held_at_customs'].value_counts(normalize=True))
print(f"\nHold reason distribution:")
print(df['hold_reason'].value_counts())
print(f"\nFirst 5 rows:")
print(df.head())