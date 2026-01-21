import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from model import BreastCancerModel, train_and_save_model

app = Flask(__name__)

# --- Model Configuration ---
predictor = BreastCancerModel()

def initialize_system():
    """Ensure model is loaded and ready before server starts."""
    if not predictor.load_model():
        print("Model state not found. Initiating training sequence...")
        train_and_save_model()
        predictor.load_model()

initialize_system()

# --- Routes ---

@app.route('/')
def index():
    """Serves the primary user interface."""
    return render_template('index.html', feature_names=predictor.feature_names)


@app.route('/predict', methods=['POST'])
def handle_prediction():
    """Processes incoming feature sets and returns diagnostic insights."""
    try:
        payload = request.get_json() or {}
        cleaned_inputs = {}

        # 1. Validation Loop
        for name in predictor.feature_names:
            val = payload.get(name)
            
            if val is None:
                return jsonify({'success': False, 'error': f'Missing: {name}'}), 400
            
            try:
                num_val = float(val)
                if num_val < 0:
                    raise ValueError
                cleaned_inputs[name] = num_val
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': f'Invalid numeric value for {name}'}), 400

        # 2. Model Inference
        label_code, prob = predictor.predict(cleaned_inputs)
        
        # Mapping: 0 = Malignant, 1 = Benign
        is_benign = bool(label_code == 1)
        confidence_score = prob if is_benign else (1 - prob)
        
        # 3. Response Construction
        return jsonify({
            'success': True,
            'diagnosis': 'Benign' if is_benign else 'Malignant',
            'is_benign': is_benign,
            'confidence': round(confidence_score * 100, 2),
            'message': '✓ Analysis suggests a BENIGN tumor.' if is_benign 
                       else '⚠️ Analysis suggests a MALIGNANT tumor.',
            'disclaimer': 'Educational tool only. Consult a medical professional for clinical diagnosis.'
        })

    except Exception as err:
        return jsonify({'success': False, 'error': str(err)}), 500


@app.route('/sample-data')
def get_test_samples():
    """Retrieves representative data points for UI testing."""
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'breast_cancer.csv')
        df = pd.read_csv(data_path)

        # Extraction logic
        samples = {
            'benign_sample': df[df['diagnosis'] == 1].iloc[0].drop('diagnosis').to_dict(),
            'malignant_sample': df[df['diagnosis'] == 0].iloc[0].drop('diagnosis').to_dict()
        }
        
        return jsonify({'success': True, **samples})
    
    except Exception as e:
        return jsonify({'success': False, 'error': f"Data retrieval failed: {str(e)}"}), 500


if __name__ == '__main__':
    # Local execution parameters
    app.run(host='0.0.0.0', port=5000, debug=True)