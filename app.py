from flask import Flask, render_template, Response, jsonify, request
import gunicorn
from camera import *
import os
import base64
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from backend.history import add_history_entry, add_liked_song, get_profile_data
from backend.recommender import recommend_songs_ml

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

headings = ("Name","Album","Artist","Type")
df1 = music_rec()
df1 = df1.head(15)

# Global camera variable
camera_stream = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_camera():
    """Get or create camera stream"""
    global camera_stream
    if camera_stream is None or not camera_stream.isOpened():
        try:
            camera_stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not camera_stream.isOpened():
                # Try default backend
                camera_stream = cv2.VideoCapture(0)
            camera_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception as e:
            print(f"Error opening camera: {e}")
            camera_stream = None
    return camera_stream

def release_camera():
    """Release camera resource"""
    global camera_stream
    if camera_stream is not None:
        camera_stream.release()
        camera_stream = None

def gen_frames():
    """Generate camera frames for streaming"""
    while True:
        cam = get_camera()
        if cam is None:
            # Return black frame with error message
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            continue
        
        success, frame = cam.read()
        if not success:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Failed to read frame", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Resize for display
            frame = cv2.resize(frame, (640, 480))
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html', headings=headings, data=df1)

@app.route('/video_feed')
def video_feed():
    """Stream camera feed"""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture_image():
    """Capture a frame from the camera and process it"""
    try:
        cam = get_camera()
        if cam is None:
            return jsonify({'error': 'Camera not available'}), 500
        
        success, frame = cam.read()
        if not success:
            return jsonify({'error': 'Failed to capture frame from camera'}), 500
        
        # Save captured frame temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_temp.jpg')
        cv2.imwrite(temp_path, frame)
        
        # Process image to detect emotion
        emotion_name, emotion_index, image_bytes, recommendations_df, songs_payload = detect_emotion_from_image(temp_path)
        
        # Convert image bytes to base64 for JSON response
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Convert recommendations to JSON
        recommendations = recommendations_df.to_dict('records')
        songs = [
            {
                'Song': song.get('Song'),
                'Artist': song.get('Artist'),
                'Album': song.get('Album', song.get('MoodLabel', '-')),
                'Type': song.get('Type', song.get('Genre', ''))
            }
            for song in songs_payload
        ]
        try:
            add_history_entry(emotion_name, songs, image_base64)
        except Exception as history_error:
            print(f"Failed to store history entry: {history_error}")
        
        # Clean up temporary file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return jsonify({
            'success': True,
            'emotion': emotion_name,
            'emotion_index': emotion_index,
            'image': image_base64,
            'recommendations': recommendations,
            'songs': songs
        })
    except Exception as e:
        return jsonify({'error': f'Error capturing image: {str(e)}'}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop and release camera"""
    release_camera()
    return jsonify({'success': True, 'message': 'Camera released'})

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image to detect emotion
            emotion_name, emotion_index, image_bytes, recommendations_df, songs_payload = detect_emotion_from_image(filepath)
            
            # Convert image bytes to base64 for JSON response
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Convert recommendations to JSON
            recommendations = recommendations_df.to_dict('records')
            songs = [
                {
                    'Song': song.get('Song'),
                    'Artist': song.get('Artist'),
                    'Album': song.get('Album', song.get('MoodLabel', '-')),
                    'Type': song.get('Type', song.get('Genre', ''))
                }
                for song in songs_payload
            ]
            try:
                add_history_entry(emotion_name, songs, image_base64)
            except Exception as history_error:
                print(f"Failed to store history entry: {history_error}")
            
            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass
            
            return jsonify({
                'success': True,
                'emotion': emotion_name,
                'emotion_index': emotion_index,
                'image': image_base64,
            'recommendations': recommendations,
            'songs': songs
            })
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload an image (png, jpg, jpeg, gif, bmp)'}), 400

@app.route('/profile')
def profile():
    history, personal_emotion, liked_songs = get_profile_data()
    personalized_songs = []
    if personal_emotion:
        try:
            personalized_songs = recommend_songs_ml(personal_emotion)
        except Exception as rec_error:
            print(f"Failed to load personalized recommendations: {rec_error}")
    return render_template(
        'profile.html',
        history=history,
        personal_emotion=personal_emotion,
        personal_recommendations=personal_songs_format(personalized_songs),
        liked_songs=liked_songs
    )

def personal_songs_format(songs_payload):
    if not songs_payload:
        return []
    formatted = []
    for song in songs_payload:
        formatted.append(
            {
                'Song': song.get('Song'),
                'Artist': song.get('Artist'),
                'Album': song.get('Album', song.get('MoodLabel', '-')),
                'Type': song.get('Type', song.get('Genre', ''))
            }
        )
    return formatted

@app.route('/like_song', methods=['POST'])
def like_song():
    try:
        payload = request.get_json(force=True)
        song_data = payload.get('song')
        if not song_data:
            return jsonify({'error': 'Invalid payload'}), 400
        add_liked_song(song_data)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': f'Failed to save like: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("Starting Emotion Music Recommendation Server")
    print("=" * 50)
    print(f"Server will be accessible at:")
    print(f"  - Local: http://localhost:5000")
    print(f"  - Network: http://0.0.0.0:5000")
    print("=" * 50)
    app.debug = True
    app.run(host='0.0.0.0', port=5000, threaded=True)
