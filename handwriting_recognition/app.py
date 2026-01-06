# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import io
import base64
import os
import pickle
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Buat folder uploads jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('model', exist_ok=True)

# Load model dan labels
print("Memuat model CNN...")
try:
    model = keras.models.load_model('model/model.h5')
    with open('model/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open('model/preprocess_info.pkl', 'rb') as f:
        preprocess_info = pickle.load(f)
    print(f"✅ Model berhasil dimuat. Labels: {labels}")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("⚠️ Pastikan model sudah di-training terlebih dahulu")
    model = None
    labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
              'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def preprocess_image(image, img_size=28):
    """Preprocess image untuk model CNN"""
    # Convert to grayscale jika perlu
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize ke 28x28
    image = cv2.resize(image, (img_size, img_size))
    
    # Invert colors (background putih, tulisan hitam)
    image = cv2.bitwise_not(image)
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Add channel dimension
    image = np.expand_dims(image, axis=-1)
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_image(image):
    """Memprediksi gambar"""
    if model is None:
        return None, None, "Model belum dimuat"
    
    try:
        # Preprocess
        processed_img = preprocess_image(image)
        
        # Predict
        predictions = model.predict(processed_img, verbose=0)
        
        # Get top predictions
        top_indices = np.argsort(predictions[0])[::-1][:5]
        top_labels = [labels[i] for i in top_indices]
        top_confidences = [float(predictions[0][i]) * 100 for i in top_indices]
        
        # Predicted class
        predicted_idx = np.argmax(predictions[0])
        predicted_label = labels[predicted_idx]
        confidence = float(predictions[0][predicted_idx]) * 100
        
        # Prepare result
        result = {
            'predicted': predicted_label,
            'confidence': round(confidence, 2),
            'all_predictions': list(zip(top_labels, top_confidences)),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result, processed_img, None
        
    except Exception as e:
        return None, None, str(e)

@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html', labels=labels)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint untuk prediksi"""
    try:
        # Cek apakah ada file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Read image file
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes))
                image = np.array(image)
                
                # Save uploaded file
                filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
        # Cek apakah ada canvas data
        elif 'canvas_data' in request.json:
            # Get base64 image from canvas
            canvas_data = request.json['canvas_data']
            
            # Remove header
            if ',' in canvas_data:
                canvas_data = canvas_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(canvas_data)
            image = Image.open(io.BytesIO(image_bytes))
            image = np.array(image)
            
            # Create filename
            filename = f"canvas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Predict
        result, processed_img, error = predict_image(image)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Tambahkan filename ke result
        result['filename'] = filename
        
        # Simpan processed image
        if processed_img is not None:
            processed_filename = f"processed_{filename}"
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            
            # Save processed image
            processed_img_display = (processed_img[0].squeeze() * 255).astype(np.uint8)
            cv2.imwrite(processed_filepath, processed_img_display)
            result['processed_filename'] = processed_filename
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/delete_upload', methods=['POST'])
def delete_upload():
    """Delete an uploaded file from the server. Expects JSON: {"filename": "..."} """
    try:
        data = request.get_json(force=True)
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'filename required'}), 400

        # Prevent path traversal
        safe_name = os.path.basename(filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)

        if not os.path.exists(file_path):
            return jsonify({'error': 'file not found'}), 404

        os.remove(file_path)
        return jsonify({'status': 'deleted', 'filename': safe_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/model_info')
def model_info():
    """Get model information"""
    info = {
        'labels': labels,
        'input_shape': model.input_shape if model else None,
        'model_loaded': model is not None,
        'total_labels': len(labels),
        'preprocess_info': preprocess_info if 'preprocess_info' in locals() else None
    }
    return jsonify(info)

@app.route('/clear_canvas', methods=['POST'])
def clear_canvas():
    """Clear canvas handler"""
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Aplikasi Pengenalan Tulisan Tangan A-Z")
    print("="*50)
    print("📂 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("🔗 Local URL: http://localhost:5000")
    print("="*50)
    
    # Jalankan server
    app.run(debug=True, host='0.0.0.0', port=5000)