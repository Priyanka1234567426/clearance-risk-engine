"""
Rules Engine — deterministic lookups for clearance requirements.

Given an HS code and destination country, returns:
  - required documents
  - duty rate %
  - restrictions

In production this would query a database that's continuously updated by
ingestion jobs from CBP HTSUS, UK Global Tariff, EU TARIC, etc.
For this project, we read from the synthetic rules_database.csv.
"""

import pandas as pd


class RulesEngine:
    """Loads the rules database once and serves lookups."""

    def __init__(self, rules_path='data/rules_database.csv'):
        self.rules_df = pd.read_csv(rules_path)
        # Build a fast lookup index: (hs_code, country) → row
        self.rules_df['hs_code'] = self.rules_df['hs_code'].astype(str)
        self._index = self.rules_df.set_index(['hs_code', 'destination_country'])

    def lookup(self, hs_code: str, destination_country: str) -> dict:
        """
        Look up the rules for one (hs_code, country) combination.

        Returns a dict with required_docs (list), duty_rate_pct, restrictions, source.
        Raises KeyError if the combination doesn't exist.
        """
        try:
            row = self._index.loc[(str(hs_code), destination_country)]
        except KeyError:
            raise KeyError(
                f"No rule found for hs_code={hs_code}, destination={destination_country}"
            )

        return {
            'hs_code': hs_code,
            'hs_description': row['hs_description'],
            'destination_country': destination_country,
            'required_docs': row['required_docs'].split(';'),
            'duty_rate_pct': float(row['duty_rate_pct']),
            'restrictions': row['restrictions'] if pd.notna(row['restrictions']) else 'None',
            'source': row['source'],
            'effective_date': row['effective_date'],
        }


# Quick test when run directly
if __name__ == '__main__':
    engine = RulesEngine()

    test_cases = [
        ('61091000', 'US'),
        ('61091000', 'UK'),
        ('62121000', 'US'),
        ('61101100', 'DE'),
    ]

    for hs_code, country in test_cases:
        result = engine.lookup(hs_code, country)
        print(f"\n{'='*60}")
        print(f"HS code: {hs_code} ({result['hs_description']})")
        print(f"Destination: {country}")
        print(f"Required documents:")
        for doc in result['required_docs']:
            print(f"  - {doc}")
        print(f"Duty rate: {result['duty_rate_pct']}%")
        print(f"Restrictions: {result['restrictions']}")
        print(f"Source: {result['source']} (effective {result['effective_date']})")