from backend.predictor import predict_language
import json

print("Testing English / Latin script")
res1 = predict_language("This is a simple english sentence.")
print(json.dumps(res1, indent=2))

print("\nTesting Hindi with ambiguity trigger")
# 'है' falls into Devanagari range. Since Hindi/Bhojpuri both use it, 
# our mock model loader forces an ambiguous probability split to trigger the rule calculation natively.
res2 = predict_language("यह एक वाक्य है")
print(json.dumps(res2, indent=2))
