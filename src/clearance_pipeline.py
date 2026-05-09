"""
End-to-End Clearance Pipeline
Takes consignor input → predicts HS code → looks up rules → computes duty → generates message.
This is the integration of the ML model with the deterministic rules engine.
"""

import joblib
import json
import numpy as np
from rules_engine import RulesEngine


# ====================================================================
# FX rates (in production: ingested daily from RBI / OANDA / XE API)
# Hardcoded here for the synthetic project; treated as a reference table.
# ====================================================================

FX_RATES_TO_USD = {
    'INR': 1 / 83,
    'CNY': 1 / 7,
    'BDT': 1 / 110,
    'VND': 1 / 25000,
    'USD': 1.0,
}


class ClearancePipeline:
    """End-to-end pipeline for proactive clearance assistance."""

    def __init__(self,
                 model_path='artifacts/hs_code_model.pkl',
                 vectorizer_path='artifacts/tfidf_vectorizer.pkl',
                 hs_descriptions_path='artifacts/hs_descriptions.json',
                 rules_path='data/rules_database.csv'):
        # Load ML artifacts
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        with open(hs_descriptions_path) as f:
            self.hs_descriptions = json.load(f)

        # Load rules engine
        self.rules_engine = RulesEngine(rules_path)

    def predict_hs_codes(self, description: str, top_k: int = 3):
        """Step 1: Predict top-K HS codes from product description."""
        X = self.vectorizer.transform([description])
        probs = self.model.predict_proba(X)[0]
        top_indices = np.argsort(probs)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            hs_code = self.model.classes_[idx]
            results.append({
                'hs_code': hs_code,
                'hs_description': self.hs_descriptions.get(hs_code, 'Unknown'),
                'confidence': float(probs[idx]),
            })
        return results

    def convert_to_usd(self, amount: float, currency: str) -> float:
        """Step 2: Convert local currency to USD using ingested FX rates."""
        if currency not in FX_RATES_TO_USD:
            raise ValueError(f"Unsupported currency: {currency}")
        return round(amount * FX_RATES_TO_USD[currency], 2)

    def calculate_duty(self, value_usd: float, duty_rate_pct: float) -> float:
        """Step 3: Calculate duty in USD."""
        return round(value_usd * duty_rate_pct / 100, 2)

    def generate_message(self, prediction, rule, value_usd, duty_amount,
                         declared_value_local, currency):
        """Step 4: Generate the consignee message (template-based)."""
        confidence = prediction['confidence']
        hs_code = prediction['hs_code']
        hs_desc = prediction['hs_description']

        docs_list = '\n  '.join(f"- {doc}" for doc in rule['required_docs'])

        # Confidence-routed message
        if confidence >= 0.90:
            classification_line = f"Classified as: {hs_code} — {hs_desc}"
        elif confidence >= 0.60:
            classification_line = (
                f"Likely classified as: {hs_code} — {hs_desc} (confidence {confidence:.0%}). "
                "Please confirm at upload."
            )
        else:
            classification_line = (
                f"Classification uncertain (top guess: {hs_code} — {hs_desc} at {confidence:.0%}). "
                "We may need a follow-up question to confirm."
            )

        restrictions_line = ""
        if rule['restrictions'] and rule['restrictions'] != 'None':
            restrictions_line = f"\n⚠️  Note: {rule['restrictions']}"

        message = (
            f"Hi! Your shipment is on its way to {rule['destination_country']}.\n\n"
            f"{classification_line}\n\n"
            f"To clear customs at destination, please upload these documents:\n"
            f"  {docs_list}\n\n"
            f"Declared value: {declared_value_local:,.2f} {currency} "
            f"(≈ ${value_usd:,.2f} USD)\n"
            f"Estimated duty ({rule['duty_rate_pct']}%): ${duty_amount:,.2f} USD"
            f"{restrictions_line}\n\n"
            f"Source: {rule['source']} (effective {rule['effective_date']})"
        )
        return message

    def process_shipment(self, description: str, destination_country: str,
                         declared_value_local: float, currency: str):
        """Run the full end-to-end pipeline for one shipment."""

        # Step 1: Predict HS codes
        predictions = self.predict_hs_codes(description, top_k=3)
        top_prediction = predictions[0]

        # Step 2: Look up rules for the top prediction
        rule = self.rules_engine.lookup(top_prediction['hs_code'], destination_country)

        # Step 3: Convert currency
        value_usd = self.convert_to_usd(declared_value_local, currency)

        # Step 4: Calculate duty
        duty_amount = self.calculate_duty(value_usd, rule['duty_rate_pct'])

        # Step 5: Generate message
        message = self.generate_message(
            top_prediction, rule, value_usd, duty_amount,
            declared_value_local, currency,
        )

        return {
            'predictions': predictions,
            'rule': rule,
            'value_usd': value_usd,
            'duty_amount_usd': duty_amount,
            'consignee_message': message,
        }


# ====================================================================
# Demo when run directly
# ====================================================================

if __name__ == '__main__':
    pipeline = ClearancePipeline()

    test_shipments = [
        {
            'description': "Men's blue cotton t-shirt, size M",
            'destination_country': 'US',
            'declared_value_local': 4150.0,  # INR
            'currency': 'INR',
        },
        {
            'description': "merino wool sweater for men",
            'destination_country': 'DE',
            'declared_value_local': 8300.0,  # INR
            'currency': 'INR',
        },
        {
            'description': "Women's cotton dress, navy",
            'destination_country': 'UK',
            'declared_value_local': 700.0,  # CNY
            'currency': 'CNY',
        },
    ]

    for i, shipment in enumerate(test_shipments, 1):
        print(f"\n{'='*70}")
        print(f"SHIPMENT {i}")
        print(f"{'='*70}")
        print(f"Input: '{shipment['description']}'")
        print(f"  → {shipment['declared_value_local']:,.0f} {shipment['currency']} "
              f"to {shipment['destination_country']}")

        result = pipeline.process_shipment(**shipment)

        print(f"\n--- Top-3 HS code predictions ---")
        for j, pred in enumerate(result['predictions'], 1):
            print(f"  {j}. [{pred['hs_code']}] {pred['hs_description']} "
                  f"({pred['confidence']:.1%})")

        print(f"\n--- Generated consignee message ---")
        print(result['consignee_message'])