from flask import Flask, render_template, request, jsonify, send_file, session, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
from tart_verifier import TartVerifier
import base64
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = 'your-secret-key-here'  # Required for session management

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize TartVerifier
tart_verifier = TartVerifier()

@app.route('/')
def index():
    # Reset session on new visit
    session['admin_image_set'] = False
    return render_template('index.html', initial_mode='admin')
@app.route('/upload', methods=['POST'])
def upload_file():
    print("Starting upload_file endpoint")
    if 'file' not in request.files:
        print("No file found in request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    user_type = request.form.get('userType', 'user')
    print(f"User type: {user_type}")
    
    if file.filename == '':
        print("Empty filename received")
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            print(f"Processing file: {file.filename}")
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to: {filepath}")
            
            # Handle admin image
            if user_type == 'admin':
                print("Processing admin image")
                # Save a copy as admin_image.jpg for reference
                admin_image_path = os.path.join('static', 'admin_image.jpg')
                shutil.copy2(filepath, admin_image_path)
                print(f"Admin image copied to: {admin_image_path}")
                
                tart_verifier.set_admin_image(filepath)
                session['admin_image_set'] = True
                print("Admin image successfully set")
                return jsonify({
                    'message': 'Admin image uploaded successfully',
                    'adminImageSet': True
                })
            # Handle user image
            else:
                print("Processing user image")
                if not session.get('admin_image_set'):
                    print("Error: Admin image not set")
                    return jsonify({'error': 'Please set admin image first'}), 400
                
                try:
                    print("Setting user image in verifier")
                    tart_verifier.set_user_image(filepath)
                    print("Comparing images")
                    result = tart_verifier.compare_images()
                    
                    # Ensure result is valid
                    if not result or not isinstance(result, tuple) or len(result) != 2:
                        print(f"Invalid comparison result: {result}")
                        raise ValueError("Invalid comparison result")
                    
                    all_match, position_matches = result
                    print(f"Comparison results - All match: {all_match}, Position matches: {position_matches}")
                    
                    # Ensure the comparison image exists
                    result_image_path = 'tart_comparison_result.jpg'
                    if not os.path.exists(result_image_path):
                        print(f"Result image not found at: {result_image_path}")
                        raise FileNotFoundError("Comparison result image not found")
                    print(f"Result image found at: {result_image_path}")
                    
                    return jsonify({
                        'success': True,
                        'all_match': all_match,
                        'position_matches': position_matches,
                        'result_image': 'tart_comparison_result.jpg'
                    })
                    
                except Exception as e:
                    print(f"Error in comparison: {str(e)}")
                    print(f"Exception type: {type(e)}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    return jsonify({'error': f'Failed to process comparison: {str(e)}'}), 400
                
        except Exception as e:
            print(f"Error processing upload: {str(e)}")
            print(f"Exception type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/capture', methods=['POST'])
def capture_image():
    image_data = request.json.get('image')
    user_type = request.json.get('userType', 'user')
    
    if not image_data:
        return jsonify({'error': 'No image data received'}), 400
    
    # Convert base64 image to file
    try:
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
    except Exception as e:
        return jsonify({'error': 'Invalid image data'}), 400
    
    filename = f'{user_type}_capture_{os.urandom(4).hex()}.jpg'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    with open(filepath, 'wb') as f:
        f.write(image_bytes)
    
    # Handle admin image
    if user_type == 'admin':
        tart_verifier.set_admin_image(filepath)
        session['admin_image_set'] = True
        return jsonify({
            'message': 'Admin image captured successfully',
            'adminImageSet': True
        })
    # Handle user image
    else:
        if not session.get('admin_image_set'):
            return jsonify({'error': 'Please set admin image first'}), 400
        
        tart_verifier.set_user_image(filepath)
        result = tart_verifier.compare_images()
        return jsonify(result)

@app.route('/check-admin-image', methods=['GET'])
def check_admin_image():
    return jsonify({'adminImageSet': session.get('admin_image_set', False)})

@app.route('/reset', methods=['POST'])
def reset_session():
    session.clear()
    return jsonify({'message': 'Session reset successfully'})

# Clean up uploaded files periodically (you might want to implement this)
def cleanup_old_files():
    # Implementation for cleaning up old uploaded files
    pass

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/result')
def result():
    return render_template('result.html')

# Add this to ensure static folder exists
import os
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

if __name__ == '__main__':
    app.run(debug=True)
