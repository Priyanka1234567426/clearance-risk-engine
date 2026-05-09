"""
Synthetic Apparel Clearance Dataset Generator
Generates 5000 fake apparel shipment records with product descriptions,
HS codes, shipment features, and clearance outcomes.
Used for the HS Code Recommender and Hold Risk Prediction notebooks.
"""

import numpy as np
import pandas as pd

# Reproducibility — same seed = same output every run
np.random.seed(42)

# Number of records to generate
N = 5000
# ====================================================================
# HS CODE TAXONOMY
# Each entry = one HS code + the attributes that describe products
# falling under that code. The generator picks a code, then samples
# attribute values from its lists to build a realistic description.
# ====================================================================

HS_CATALOG = [
    # --- Knitted apparel (Chapter 61) ---
    {'hs_code': '61091000', 'description': 'T-shirts of cotton, knitted',
     'gender': ['Men', 'Women', 'Unisex'], 'fiber': ['cotton'],
     'style': ['t-shirt', 'tee', 'singlet'], 'construction': 'knitted'},

    {'hs_code': '61099020', 'description': 'T-shirts of synthetic fibers, knitted',
     'gender': ['Men', 'Women', 'Unisex'], 'fiber': ['polyester', 'synthetic'],
     'style': ['t-shirt', 'sports tee'], 'construction': 'knitted'},

    {'hs_code': '61102000', 'description': 'Sweaters of cotton, knitted',
     'gender': ['Men', 'Women'], 'fiber': ['cotton'],
     'style': ['sweater', 'pullover', 'jumper'], 'construction': 'knitted'},

    {'hs_code': '61103000', 'description': 'Sweaters of synthetic fibers, knitted',
     'gender': ['Men', 'Women'], 'fiber': ['polyester', 'acrylic'],
     'style': ['sweater', 'pullover'], 'construction': 'knitted'},

    {'hs_code': '61101100', 'description': 'Sweaters of wool, knitted',
     'gender': ['Men', 'Women'], 'fiber': ['wool', 'merino wool'],
     'style': ['sweater', 'cardigan'], 'construction': 'knitted'},

    {'hs_code': '61051000', 'description': "Men's shirts of cotton, knitted",
     'gender': ['Men'], 'fiber': ['cotton'],
     'style': ['polo shirt', 'knit shirt'], 'construction': 'knitted'},

    {'hs_code': '61061000', 'description': "Women's blouses of cotton, knitted",
     'gender': ['Women'], 'fiber': ['cotton'],
     'style': ['blouse', 'top', 'knit shirt'], 'construction': 'knitted'},

    {'hs_code': '61046200', 'description': "Women's trousers of cotton, knitted",
     'gender': ['Women'], 'fiber': ['cotton'],
     'style': ['leggings', 'pants'], 'construction': 'knitted'},

    {'hs_code': '61046300', 'description': "Women's trousers of synthetic, knitted",
     'gender': ['Women'], 'fiber': ['polyester', 'spandex'],
     'style': ['leggings', 'yoga pants'], 'construction': 'knitted'},

    {'hs_code': '61034200', 'description': "Men's trousers of cotton, knitted",
     'gender': ['Men'], 'fiber': ['cotton'],
     'style': ['trackpants', 'sweatpants'], 'construction': 'knitted'},

    {'hs_code': '61034300', 'description': "Men's trousers of synthetic, knitted",
     'gender': ['Men'], 'fiber': ['polyester'],
     'style': ['joggers', 'sweatpants'], 'construction': 'knitted'},

    {'hs_code': '61121100', 'description': 'Tracksuits of cotton, knitted',
     'gender': ['Men', 'Women'], 'fiber': ['cotton'],
     'style': ['tracksuit', 'sweatsuit'], 'construction': 'knitted'},

    {'hs_code': '61121200', 'description': 'Tracksuits of synthetic, knitted',
     'gender': ['Men', 'Women'], 'fiber': ['polyester'],
     'style': ['tracksuit'], 'construction': 'knitted'},

    {'hs_code': '61159500', 'description': 'Socks of cotton, knitted',
     'gender': ['Men', 'Women', 'Unisex'], 'fiber': ['cotton'],
     'style': ['socks', 'crew socks'], 'construction': 'knitted'},

    {'hs_code': '61159600', 'description': 'Socks of synthetic, knitted',
     'gender': ['Men', 'Women', 'Unisex'], 'fiber': ['polyester'],
     'style': ['sports socks'], 'construction': 'knitted'},

    {'hs_code': '61082100', 'description': "Women's panties of cotton, knitted",
     'gender': ['Women'], 'fiber': ['cotton'],
     'style': ['panties', 'briefs'], 'construction': 'knitted'},

    {'hs_code': '61013000', 'description': "Men's overcoats of synthetic, knitted",
     'gender': ['Men'], 'fiber': ['polyester'],
     'style': ['overcoat', 'anorak'], 'construction': 'knitted'},

    {'hs_code': '61023000', 'description': "Women's overcoats of synthetic, knitted",
     'gender': ['Women'], 'fiber': ['polyester'],
     'style': ['overcoat', 'windbreaker'], 'construction': 'knitted'},

    # --- Woven apparel (Chapter 62) ---
    {'hs_code': '62034200', 'description': "Men's trousers of cotton, woven",
     'gender': ['Men'], 'fiber': ['cotton', 'denim'],
     'style': ['jeans', 'trousers', 'chinos'], 'construction': 'woven'},

    {'hs_code': '62034300', 'description': "Men's trousers of synthetic, woven",
     'gender': ['Men'], 'fiber': ['polyester'],
     'style': ['formal pants'], 'construction': 'woven'},

    {'hs_code': '62046200', 'description': "Women's trousers of cotton, woven",
     'gender': ['Women'], 'fiber': ['cotton', 'denim'],
     'style': ['jeans', 'trousers'], 'construction': 'woven'},

    {'hs_code': '62046300', 'description': "Women's trousers of synthetic, woven",
     'gender': ['Women'], 'fiber': ['polyester'],
     'style': ['formal pants'], 'construction': 'woven'},

    {'hs_code': '62044200', 'description': "Women's dresses of cotton, woven",
     'gender': ['Women'], 'fiber': ['cotton'],
     'style': ['dress', 'sundress'], 'construction': 'woven'},

    {'hs_code': '62044300', 'description': "Women's dresses of synthetic, woven",
     'gender': ['Women'], 'fiber': ['polyester'],
     'style': ['evening dress'], 'construction': 'woven'},

    {'hs_code': '62052000', 'description': "Men's shirts of cotton, woven",
     'gender': ['Men'], 'fiber': ['cotton'],
     'style': ['formal shirt', 'dress shirt'], 'construction': 'woven'},

    {'hs_code': '62053000', 'description': "Men's shirts of synthetic, woven",
     'gender': ['Men'], 'fiber': ['polyester'],
     'style': ['dress shirt'], 'construction': 'woven'},

    {'hs_code': '62063000', 'description': "Women's blouses of cotton, woven",
     'gender': ['Women'], 'fiber': ['cotton'],
     'style': ['blouse', 'tunic'], 'construction': 'woven'},

    {'hs_code': '62019300', 'description': "Men's jackets of synthetic, woven",
     'gender': ['Men'], 'fiber': ['polyester', 'nylon'],
     'style': ['jacket', 'windbreaker'], 'construction': 'woven'},

    {'hs_code': '62029300', 'description': "Women's jackets of synthetic, woven",
     'gender': ['Women'], 'fiber': ['polyester', 'nylon'],
     'style': ['jacket', 'windbreaker'], 'construction': 'woven'},

    {'hs_code': '62121000', 'description': 'Brassieres',
     'gender': ['Women'], 'fiber': ['cotton', 'nylon', 'polyester'],
     'style': ['bra', 'sports bra'], 'construction': 'mixed'},
]

print(f"Loaded {len(HS_CATALOG)} HS codes")
# ====================================================================
# VALUE POOLS — attributes that vary across all products
# regardless of which HS code they belong to.
# ====================================================================

COLOR_OPTIONS = ['black', 'white', 'navy', 'blue', 'red', 'grey', 'beige', 'olive', 'maroon', 'pink']
SIZE_OPTIONS = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
FIT_OPTIONS = ['', 'casual fit', 'slim fit', 'regular fit', 'premium quality', 'export quality', '']

# ====================================================================
# DESCRIPTION GENERATOR
# Takes one HS code template and returns one free-text product description.
# Uses 5 different sentence templates so descriptions look messy/varied,
# like real consignors writing in their own style.
# ====================================================================

def generate_description(item):
    """Generate one realistic product description from an HS code template."""
    gender = np.random.choice(item['gender'])
    fiber = np.random.choice(item['fiber'])
    style = np.random.choice(item['style'])
    color = np.random.choice(COLOR_OPTIONS)
    size = np.random.choice(SIZE_OPTIONS)
    fit = np.random.choice(FIT_OPTIONS)
    construction = item['construction']

    # 5 templates — random pick adds variety
    templates = [
        f"{gender}'s {fiber} {style}, {color}, size {size}",
        f"{fiber} {style} for {gender.lower()}, {color} color",
        f"{gender} {color} {style} ({fiber}, {construction})",
        f"{style} - {gender}'s, 100% {fiber}, {color}",
        f"{color} {fiber} {style} {gender.lower()}'s wear",
    ]
    desc = np.random.choice(templates)

    if fit:
        desc += f", {fit}"
    return desc


# Quick test — generate 5 sample descriptions to see if it works
print("\nSample descriptions from generator:")
for _ in range(5):
    item = np.random.choice(HS_CATALOG)
    print(f"  [{item['hs_code']}]  {generate_description(item)}")
    # ====================================================================
# MAIN GENERATION LOOP
# Build N shipment records, one at a time, and collect them in a list.
# ====================================================================

records = []

for i in range(N):
    # 1. Pick a random HS code template
    item = np.random.choice(HS_CATALOG)

    # 2. Generate the product description
    description = generate_description(item)

    # 3. Pick consignor country first; currency follows from it
    consignor_country = np.random.choice(['IN', 'CN', 'BD', 'VN'], p=[0.50, 0.30, 0.10, 0.10])
    currency_map = {'IN': 'INR', 'CN': 'CNY', 'BD': 'BDT', 'VN': 'VND'}
    fx_multiplier = {'IN': 83, 'CN': 7, 'BD': 110, 'VN': 25000}  # approx local per USD
    currency = currency_map[consignor_country]

    # Generate USD-equivalent value, then express in local currency
    base_usd = np.random.lognormal(5.5, 1.0)
    declared_value_local = np.round(base_usd * fx_multiplier[consignor_country], 2)

    # 4. Build the full record
    record = {
        'shipment_id': f'APP{i:06d}',
        'product_description': description,
        'hs_code': item['hs_code'],
        'hs_description': item['description'],
        'consignor_country': consignor_country,
        'consignee_country': np.random.choice(['US', 'UK', 'CA', 'DE'], p=[0.65, 0.15, 0.10, 0.10]),
        'declared_value_local': declared_value_local,
        'currency': currency,
        'weight_kg': np.round(np.random.lognormal(1.0, 0.8), 2),
        'num_line_items': np.random.choice([1, 2, 3, 4, 5, 8], p=[0.40, 0.25, 0.15, 0.10, 0.06, 0.04]),
        'consignor_history_score': float(np.clip(np.round(np.random.normal(75, 18)), 0, 100)),
        'is_first_time_consignee': np.random.choice([0, 1], p=[0.70, 0.30]),
        'documentation_completeness_score': float(np.clip(np.round(np.random.normal(82, 15)), 0, 100)),
    }
    records.append(record)

# 4. Convert the list of dicts into a pandas DataFrame
df = pd.DataFrame(records)

# 5. Add a derived column: lane (origin-destination)
df['lane'] = df['consignor_country'] + '-' + df['consignee_country']

print(f"\nGenerated {len(df)} records")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3))
# Save to CSV
df.to_csv('data/apparel_shipments.csv', index=False)
print(f"\nSaved to data/apparel_shipments.csv")