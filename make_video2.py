import cv2
cap = cv2.VideoCapture('data/demo_video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter('data/video2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
while True:
    ret, frame = cap.read()
    if not ret: break
    out.write(frame)
cap.release()
out.release()
print('done')
