import urllib.request, os
os.makedirs('checkpoints/mediapipe', exist_ok=True)
url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
print('Downloading...')
urllib.request.urlretrieve(url, 'checkpoints/mediapipe/face_landmarker.task')
print('size:', os.path.getsize('checkpoints/mediapipe/face_landmarker.task'))
