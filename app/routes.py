from flask import render_template, request, jsonify
from flask_cors import CORS
from app import app
from app.models.scam_detector import ScamDetector
import traceback

CORS(app)

scam_detector = ScamDetector()
print(f"ScamDetector initialized with {len(scam_detector.train_data)} training samples")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/debug')
def debug():
    return render_template('debug.html')

@app.route('/phishing')
def phishing():
    return render_template('phishing.html')

@app.route('/impersonation')
def impersonation():
    return render_template('impersonation.html')

@app.route('/financial')
def financial():
    return render_template('financial.html')

@app.route('/urgent')
def urgent():
    return render_template('urgent.html')

@app.route('/api/detect', methods=['POST'])
def detect_scam():
    try:
        data = request.json
        message = data.get('message')
        if not message:
            return jsonify({'error': 'No message provided'}), 400
         
        print(f"\n=== API DETECT CALLED ===")
        print(f"Message: {message}")
        
        result = scam_detector.predict(message)
        
        print(f"API returning result:")
        print(f"scam_type: {result.get('scam_type')} (type: {type(result.get('scam_type'))})")
        print(f"confidence: {result.get('confidence')}")
        print(f"suspicious_ngrams count: {len(result.get('suspicious_ngrams', []))}")
        print(f"=== END API DETECT ===\n")
        
        return jsonify(result)
    except Exception as e:
        print(f"Error in detect_scam: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        message = data.get('message')
        is_scam = data.get('is_scam')
        scam_type = data.get('scam_type')
        platform = data.get('platform')
         
        if not all([message, is_scam is not None, scam_type, platform]):
            return jsonify({'error': 'Missing required fields'}), 400
         
        result = scam_detector.add_feedback(message, is_scam, scam_type, platform)
        return jsonify(result)
    except Exception as e:
        print(f"Error in submit_feedback: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        stats = scam_detector.get_stats()
        return jsonify(stats)
    except Exception as e:
        print(f"Error in get_stats: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test_connection():
    return jsonify({'status': 'success', 'message': 'Backend connection successful'})
