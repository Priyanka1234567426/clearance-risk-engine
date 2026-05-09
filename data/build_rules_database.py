"""
Synthetic Rules Database Generator
For each (HS code, destination country) pair, generates plausible:
  - required documents
  - duty rate %
  - restrictions

In production, this table would be populated by ingestion jobs from
CBP HTSUS, UK Tariff, EU TARIC, etc. Here we synthesize realistic values.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# ====================================================================
# Re-use the same 30 HS codes from the shipments generator
# (We import them as a simple list of dicts here.)
# ====================================================================

HS_CODES = [
    # Knitted (Chapter 61)
    ('61091000', 'T-shirts of cotton, knitted', 'tops'),
    ('61099020', 'T-shirts of synthetic, knitted', 'tops'),
    ('61102000', 'Sweaters of cotton, knitted', 'tops'),
    ('61103000', 'Sweaters of synthetic, knitted', 'tops'),
    ('61101100', 'Sweaters of wool, knitted', 'tops'),
    ('61051000', "Men's shirts of cotton, knitted", 'tops'),
    ('61061000', "Women's blouses of cotton, knitted", 'tops'),
    ('61046200', "Women's trousers of cotton, knitted", 'bottoms'),
    ('61046300', "Women's trousers of synthetic, knitted", 'bottoms'),
    ('61034200', "Men's trousers of cotton, knitted", 'bottoms'),
    ('61034300', "Men's trousers of synthetic, knitted", 'bottoms'),
    ('61121100', 'Tracksuits of cotton, knitted', 'sets'),
    ('61121200', 'Tracksuits of synthetic, knitted', 'sets'),
    ('61159500', 'Socks of cotton, knitted', 'accessories'),
    ('61159600', 'Socks of synthetic, knitted', 'accessories'),
    ('61082100', "Women's panties of cotton, knitted", 'undergarments'),
    ('61013000', "Men's overcoats of synthetic, knitted", 'outerwear'),
    ('61023000', "Women's overcoats of synthetic, knitted", 'outerwear'),
    # Woven (Chapter 62)
    ('62034200', "Men's trousers of cotton, woven", 'bottoms'),
    ('62034300', "Men's trousers of synthetic, woven", 'bottoms'),
    ('62046200', "Women's trousers of cotton, woven", 'bottoms'),
    ('62046300', "Women's trousers of synthetic, woven", 'bottoms'),
    ('62044200', "Women's dresses of cotton, woven", 'tops'),
    ('62044300', "Women's dresses of synthetic, woven", 'tops'),
    ('62052000', "Men's shirts of cotton, woven", 'tops'),
    ('62053000', "Men's shirts of synthetic, woven", 'tops'),
    ('62063000', "Women's blouses of cotton, woven", 'tops'),
    ('62019300', "Men's jackets of synthetic, woven", 'outerwear'),
    ('62029300', "Women's jackets of synthetic, woven", 'outerwear'),
    ('62121000', 'Brassieres', 'undergarments'),
]

# ====================================================================
# Country-specific rule profiles
# Real customs regimes have different baseline document requirements.
# Each country has its own structure — these match real-world patterns.
# ====================================================================

COUNTRY_RULES = {
    'US': {
        'base_docs': ['Govt ID', 'Address Proof', 'Commercial Invoice'],
        'extra_docs_outerwear': ['Care Label Cert'],
        'extra_docs_undergarments': ['FDA Notification'],  # cosmetic-textile overlap
        'duty_range': (10.0, 20.0),  # apparel duty in US is high
        'source': 'CBP HTSUS',
    },
    'UK': {
        'base_docs': ['Govt ID', 'Address Proof', 'Origin Certificate'],
        'extra_docs_outerwear': [],
        'extra_docs_undergarments': [],
        'duty_range': (8.0, 12.0),  # UK post-Brexit standard
        'source': 'UK Global Tariff',
    },
    'CA': {
        'base_docs': ['Govt ID', 'Address Proof', 'Commercial Invoice'],
        'extra_docs_outerwear': ['Textile Labelling Compliance'],
        'extra_docs_undergarments': [],
        'duty_range': (16.0, 18.0),  # Canada apparel duty is uniform high
        'source': 'CBSA Customs Tariff',
    },
    'DE': {  # represents EU
        'base_docs': ['Govt ID', 'Address Proof', 'EORI Number', 'Origin Certificate'],
        'extra_docs_outerwear': ['REACH Compliance'],
        'extra_docs_undergarments': [],
        'duty_range': (10.0, 12.0),
        'source': 'EU TARIC',
    },
}

# ====================================================================
# Generate one rule per (HS code, destination country) pair
# ====================================================================

rules = []

for hs_code, hs_desc, category in HS_CODES:
    for country, profile in COUNTRY_RULES.items():
        # Build the document list for this combination
        docs = list(profile['base_docs'])  # copy
        if category == 'outerwear':
            docs.extend(profile['extra_docs_outerwear'])
        if category == 'undergarments':
            docs.extend(profile['extra_docs_undergarments'])

        # Wool gets origin cert globally (anti-dumping / origin sensitive)
        if 'wool' in hs_desc.lower() and 'Origin Certificate' not in docs:
            docs.append('Origin Certificate')

        # Synthetic apparel — many countries require fabric composition declaration
        if 'synthetic' in hs_desc.lower():
            docs.append('Fabric Composition Declaration')

        # Compute duty rate within country range
        duty_low, duty_high = profile['duty_range']
        duty_rate = round(np.random.uniform(duty_low, duty_high), 1)

        # Restrictions
        restrictions = 'None'
        if category == 'undergarments' and country == 'US':
            restrictions = 'FDA notification required'

        rules.append({
            'hs_code': hs_code,
            'hs_description': hs_desc,
            'destination_country': country,
            'required_docs': ';'.join(docs),
            'duty_rate_pct': duty_rate,
            'restrictions': restrictions,
            'effective_date': '2024-01-01',
            'source': profile['source'],
        })

df_rules = pd.DataFrame(rules)
df_rules.to_csv('data/rules_database.csv', index=False)

print(f"Generated {len(df_rules)} rules ({len(HS_CODES)} HS codes × {len(COUNTRY_RULES)} countries)")
print(f"\nSample rules:")
print(df_rules.head(8).to_string(index=False))
print(f"\nDuty rate stats:")
print(df_rules.groupby('destination_country')['duty_rate_pct'].describe()[['mean', 'min', 'max']])
print(f"\nSaved to data/rules_database.csv")