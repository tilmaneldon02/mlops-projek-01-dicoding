import tensorflow as tf

LABEL_KEY = "Quality"
FEATURE_KEY = ['Acidity', 'Crunchiness', 'Juiciness', 'Ripeness', 'Size', 'Sweetness', 'Weight']

def _transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def preprocessing_fn(inputs):
    """Preprocess input features."""
    outputs = {}

    # Normalize numerical features
    for feature_name in FEATURE_KEY:
        outputs[_transformed_name(feature_name)] = inputs[feature_name]

    # Leave 'quality' as it is (treat it as a categorical feature)
    outputs[_transformed_name('Quality')] = inputs['Quality']

    return outputs
