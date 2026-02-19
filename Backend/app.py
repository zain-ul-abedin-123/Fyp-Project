from flask import Flask, request, jsonify
from flask_cors import CORS
import os
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    np = None
    _HAS_NUMPY = False
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Activity classes
ACTIVITY_CLASSES = ['downstairs', 'jogging', 'sitting', 'standing', 'upstairs', 'walking']

# Model will be loaded at startup
model = None
model_type = "none"

def load_model():
    """Load the trained Bi-LSTM model"""
    global model, model_type
    
    # Try TensorFlow/Keras first
    try:
        from tensorflow.keras.models import load_model as keras_load
        from tensorflow.keras.layers import Layer
        
        class Attention(Layer):
            def __init__(self, **kwargs):
                super(Attention, self).__init__(**kwargs)
            
            def build(self, input_shape):
                self.W = self.add_weight(
                    name="att_weight",
                    shape=(input_shape[-1], 1),
                    initializer="normal"
                )
                self.b = self.add_weight(
                    name="att_bias",
                    shape=(input_shape[1], 1),
                    initializer="zeros"
                )
                super(Attention, self).build(input_shape)
            
            def call(self, x):
                import tensorflow.keras.backend as K
                e = K.tanh(K.dot(x, self.W) + self.b)
                a = K.softmax(e, axis=1)
                output = x * a
                return K.sum(output, axis=1)
        
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'bilstm_attention.h5')
        
        if os.path.exists(model_path):
            model = keras_load(model_path, custom_objects={'Attention': Attention})
            model_type = "keras"
            print(f"✓ Bi-LSTM Keras model loaded successfully!")
            return True
    except ImportError:
        print("TensorFlow/Keras not available, using fallback model")
    except Exception as e:
        print(f"Could not load Keras model: {e}, using fallback")
    
    # Fallback: Use a simple rule-based model
    print("✓ Using deterministic fallback model for predictions")
    model_type = "fallback"
    return True

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Activity Recognition API running'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model_type != "none",
        'model_type': model_type
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({'classes': ACTIVITY_CLASSES})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict activity from 12 sensor features
    """
    try:
        if model is None and model_type == "none":
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        # Extract the 12 features
        feature_keys = [
            'attitude_x', 'attitude_y', 'attitude_z',
            'gravity_x', 'gravity_y', 'gravity_z',
            'rotation_x', 'rotation_y', 'rotation_z',
            'acceleration_x', 'acceleration_y', 'acceleration_z'
        ]
        
        features = []
        for key in feature_keys:
            if key not in data:
                return jsonify({'error': f'Missing sensor: {key}'}), 400
            try:
                features.append(float(data[key]))
            except (ValueError, TypeError):
                return jsonify({'error': f'Invalid value for {key}'}), 400
        
        # Get predictions based on model type
        if model_type == "keras":
            # Use Keras model (requires numpy/tensorflow available)
            X = np.array(features, dtype=np.float32).reshape(1, 12, 1)
            predictions = model.predict(X, verbose=0)[0]
        else:
            # Use fallback model (works without numpy)
            predictions = predict_fallback(features)
        
        # Get predicted activity
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        activity = ACTIVITY_CLASSES[predicted_idx]
        
        # Build all predictions dict
        all_predictions = {
            ACTIVITY_CLASSES[i]: float(predictions[i])
            for i in range(len(ACTIVITY_CLASSES))
        }
        
        return jsonify({
            'activity': activity,
            'confidence': confidence,
            'all_predictions': all_predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_fallback(features):
    """Fallback deterministic predictor that works without numpy.

    `features` can be a list or numpy array of length 12.
    Returns a list of probabilities matching `ACTIVITY_CLASSES`.
    """
    # Ensure features is a plain list
    if _HAS_NUMPY and isinstance(features, np.ndarray):
        feats = features.flatten().tolist()
    else:
        feats = list(features)

    def mean_abs(slice_list):
        vals = [abs(float(x)) for x in slice_list]
        return sum(vals) / len(vals) if vals else 0.0

    attitude = mean_abs(feats[0:3])
    gravity = mean_abs(feats[3:6])
    rotation = mean_abs(feats[6:9])
    acceleration = mean_abs(feats[9:12])

    # Initialize scores with a small floor
    scores = [0.05] * len(ACTIVITY_CLASSES)

    # Speed analysis from acceleration
    if acceleration > 0.8:
        scores[1] = 0.9   # jogging
        scores[4] = 0.5   # upstairs
        scores[5] = 0.6   # walking
    elif acceleration > 0.4:
        scores[5] = 0.8   # walking
        scores[4] = 0.4   # upstairs
    elif acceleration > 0.1:
        scores[5] = 0.6   # walking
        scores[0] = 0.4   # downstairs (light movement)
    else:
        scores[2] = 0.8   # sitting (minimal acceleration)
        scores[3] = 0.7   # standing (minimal acceleration)

    # Rotation analysis
    if rotation > 1.5:
        scores[0] += 0.4  # downstairs
        scores[4] += 0.3  # upstairs (high rotation)
    elif rotation > 0.5:
        scores[4] += 0.3  # upstairs
        scores[0] += 0.2  # downstairs

    # Attitude analysis
    if attitude > 0.5:
        scores[2] += 0.1  # sitting

    # Normalize to probabilities
    total = sum(scores)
    if total <= 0:
        # fallback uniform
        return [1.0 / len(scores)] * len(scores)
    probs = [float(s) / float(total) for s in scores]
    return probs

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Human Activity Recognition - Backend Server")
    print("="*60)
    print("Loading model...")
    load_model()
    print(f"✓ Model type: {model_type}")
    print(f"✓ Activities: {', '.join(ACTIVITY_CLASSES)}")
    print(f"✓ Server starting on http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
