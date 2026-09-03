import joblib

pipeline = joblib.load("models/pipeline.pkl")
feature_order = joblib.load("models/feature_order.pkl")

print("=" * 80)
print("Selected Columns")
print("=" * 80)

for i, col in enumerate(pipeline["selected_columns"][:80], 1):
    print(f"{i:3d}. {col}")

print("\n")
print("=" * 80)
print("Feature Order")
print("=" * 80)

for i, col in enumerate(feature_order[:80], 1):
    print(f"{i:3d}. {col}")