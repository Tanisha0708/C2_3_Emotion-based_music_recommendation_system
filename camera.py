import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pandastable import Table, TableModel
from tensorflow.keras.preprocessing import image
import datetime
from threading import Thread
# from Spotipy import *  
import time
import pandas as pd
from backend.recommender import recommend_songs_dataframe, recommend_songs_ml
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor=0.6

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1 
cap1 = None  # Initialize cap1 to None
show_text=[0]


''' Class for calculating FPS while streaming. Used this to check performance of using another thread for video streaming '''
class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0
	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self
	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()
	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1
	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()
	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()


''' Class for using another thread for video streaming to boost performance '''
class WebcamVideoStream:
    	
		def __init__(self, src=0):
			# Try different backends for better compatibility
			self.stream = None
			# Try DirectShow first (Windows)
			try:
				self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
				if not self.stream.isOpened():
					raise Exception("DirectShow failed")
			except:
				# Fallback to default backend
				try:
					self.stream = cv2.VideoCapture(src)
					if not self.stream.isOpened():
						raise Exception("Default backend failed")
				except:
					raise Exception("Could not open camera")
			
			# Set camera properties for better performance
			self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
			self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
			self.stream.set(cv2.CAP_PROP_FPS, 30)
			
			# Try to read initial frame
			(self.grabbed, self.frame) = self.stream.read()
			if not self.grabbed:
				raise Exception("Could not read frame from camera")
			
			self.stopped = False

		def start(self):
				# start the thread to read frames from the video stream
			Thread(target=self.update, args=(), daemon=True).start()
			time.sleep(1.0)  # Give thread time to initialize
			return self
			
		def update(self):
			# keep looping infinitely until the thread is stopped
			while True:
				# if the thread indicator variable is set, stop the thread
				if self.stopped:
					self.stream.release()
					return
				# otherwise, read the next frame from the stream
				(self.grabbed, self.frame) = self.stream.read()
				if not self.grabbed:
					# If frame couldn't be read, wait a bit and try again
					time.sleep(0.1)

		def read(self):
			# return the frame most recently read
			return self.frame
		def stop(self):
			# indicate that the thread should be stopped
			self.stopped = True

''' Class for reading video stream, generating prediction and recommendations '''
class VideoCamera(object):
	
	def __init__(self):
		global cap1
		# Initialize camera only once
		try:
			if cap1 is None or not hasattr(cap1, 'stream') or not cap1.stream.isOpened():
				cap1 = WebcamVideoStream(src=0).start()
		except Exception as e:
			print(f"Error initializing camera: {e}")
			cap1 = None
	
	def get_frame(self):
		global cap1
		global df1
		
		# Ensure camera is initialized
		if cap1 is None:
			try:
				cap1 = WebcamVideoStream(src=0).start()
			except Exception as e:
				print(f"Camera initialization error: {e}")
				# Return a black frame if camera fails
				image = np.zeros((480, 640, 3), dtype=np.uint8)
				cv2.putText(image, "Camera not available", (50, 240), 
						   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
				img = Image.fromarray(image)
				img = np.array(img)
				ret, jpeg = cv2.imencode('.jpg', img)
				df1 = music_rec()
				return jpeg.tobytes(), df1
		
		try:
			image = cap1.read()
			if image is None:
				raise Exception("Could not read frame")
			
			image=cv2.resize(image,(600,500))
			gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
			face_rects=face_cascade.detectMultiScale(gray,1.3,5)
			df1 = music_rec()
			for (x,y,w,h) in face_rects:
				cv2.rectangle(image,(x,y-50),(x+w,y+h+10),(0,255,0),2)
				roi_gray_frame = gray[y:y + h, x:x + w]
				# Resize to 48x48 and normalize to 0-1 range (as model was trained)
				cropped_img = cv2.resize(roi_gray_frame, (48, 48))
				cropped_img = cropped_img.astype('float32') / 255.0  # Normalize
				cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
				prediction = emotion_model.predict(cropped_img, verbose=0)

				maxindex = int(np.argmax(prediction))
				show_text[0] = maxindex 
				#print("===========================================",music_dist[show_text[0]],"===========================================")
				#print(df1)
				cv2.putText(image, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
				df1 = music_rec(maxindex)
				
			global last_frame1
			last_frame1 = image.copy()
			pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
			img = Image.fromarray(last_frame1)
			img = np.array(img)
			ret, jpeg = cv2.imencode('.jpg', img)
			return jpeg.tobytes(), df1
		except Exception as e:
			print(f"Error reading frame: {e}")
			# Return last frame or black frame on error
			if 'last_frame1' in globals():
				image = last_frame1.copy()
			else:
				image = np.zeros((480, 640, 3), dtype=np.uint8)
			img = Image.fromarray(image)
			img = np.array(img)
			ret, jpeg = cv2.imencode('.jpg', img)
			df1 = music_rec()
			return jpeg.tobytes(), df1

def music_rec(emotion_index=None):
	if emotion_index is None:
		emotion_index = show_text[0]

	if isinstance(emotion_index, int):
		emotion_name = emotion_dict.get(emotion_index, "Neutral")
	else:
		emotion_name = emotion_index

	df = recommend_songs_dataframe(emotion_name)
	if df.empty:
		return pd.DataFrame(columns=['Name','Album','Artist','Type'])
	return df

def detect_emotion_from_image(image_path):
	"""
	Process an uploaded image to detect emotion and return results
	Returns: (emotion_name, emotion_index, processed_image_bytes, recommendations_df, songs_payload)
	"""
	try:
		# Read the image
		image = cv2.imread(image_path)
		if image is None:
			raise Exception("Could not read image file")
		
		# Resize for display
		image = cv2.resize(image, (600, 500))
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		
		# Detect faces
		face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
		
		emotion_index = 4  # Default to Neutral
		emotion_name = "Neutral"
		
		if len(face_rects) > 0:
			# Process the first detected face
			(x, y, w, h) = face_rects[0]
			
			# Draw rectangle around face
			cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
			
			# Extract face region
			roi_gray_frame = gray[y:y + h, x:x + w]
			# Resize to 48x48 and normalize to 0-1 range (as model was trained)
			cropped_img = cv2.resize(roi_gray_frame, (48, 48))
			cropped_img = cropped_img.astype('float32') / 255.0  # Normalize
			cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0)
			
			# Predict emotion
			prediction = emotion_model.predict(cropped_img, verbose=0)
			emotion_index = int(np.argmax(prediction))
			emotion_name = emotion_dict[emotion_index]
			
			# Debug: Print prediction probabilities (can be removed later)
			print(f"Emotion prediction probabilities: {prediction[0]}")
			print(f"Detected emotion: {emotion_name} (index: {emotion_index})")
			
			# Draw emotion label on image
			cv2.putText(image, emotion_name, (x+20, y-60), 
					   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
		else:
			# No face detected
			cv2.putText(image, "No Face Detected", (50, 50), 
					   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
		
		# Get music recommendations
		song_records = recommend_songs_ml(emotion_name)
		if song_records:
			df = pd.DataFrame(song_records)
			df['Name'] = df['Song']
			df['Album'] = df.get('Album', df.get('MoodLabel', '-'))
			df['Artist'] = df['Artist']
			df['Type'] = df.get('Type', df.get('Genre', ''))
			df = df[['Name','Album','Artist','Type']]
		else:
			df = pd.DataFrame(columns=['Name','Album','Artist','Type'])
		
		# Convert image to bytes
		ret, jpeg = cv2.imencode('.jpg', image)
		image_bytes = jpeg.tobytes()
		
		return emotion_name, emotion_index, image_bytes, df, song_records
		
	except Exception as e:
		print(f"Error processing image: {e}")
		# Return error image
		error_image = np.zeros((480, 640, 3), dtype=np.uint8)
		cv2.putText(error_image, f"Error: {str(e)}", (50, 240), 
				   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		ret, jpeg = cv2.imencode('.jpg', error_image)
		df = music_rec(4)  # Default to Neutral
		return "Error", 4, jpeg.tobytes(), df, []
