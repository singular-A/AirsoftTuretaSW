import cv2
from ultralytics import YOLO
import time
import pandas as pd

#cfg
CAMERA_INDEX = 0    
MODEL_NAME = 'yolo11s-pose.pt' 
CONF_THRESHOLD = 0.5 


print(f"Načítám standardní YOLO model: {MODEL_NAME}...")
try:
    model = YOLO(MODEL_NAME)
except Exception as e:
    print(f"Chyba: {e}")
    exit()

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Nelze otevřít kameru. Zkontrolujte oprávnění terminálu.")
    exit()

print("--> Běží STANDARDNÍ YOLO. Stiskněte 'q' pro ukončení.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1) # Mirror
    frame_count += 1
    
#time track
    t_start = time.perf_counter()

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, conf=CONF_THRESHOLD)
    
    t_end = time.perf_counter()
    fps = 1 / (t_end - t_start)


    if results[0].boxes.id is not None and results[0].keypoints is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results[0].boxes.id.int().cpu().tolist()
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        
        for box, track_id, keypoints in zip(boxes, track_ids, keypoints_data):
            x1, y1, x2, y2 = box
            
            # Logika Ruce nad hlavou
            l_sh, r_sh = keypoints[5], keypoints[6]   # Ramena
            l_wr, r_wr = keypoints[9], keypoints[10]  # Zápěstí
            
            hands_up = False
            # Levá ruka (Y menší = výš)
            if l_wr[2] > 0.5 and l_sh[2] > 0.5 and l_wr[1] < l_sh[1]: hands_up = True
            # Pravá ruka
            if r_wr[2] > 0.5 and r_sh[2] > 0.5 and r_wr[1] < r_sh[1]: hands_up = True
            
            # Vizualizace
            color = (0, 0, 255) if hands_up else (0, 255, 0) # cervena =vzdava se, zelena = ve hre
            status = "OUT" if hands_up else "IN"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID:{track_id} {status}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # results
    cv2.putText(frame, f"Mode: YOLO (PyTorch) | FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow('YOLO Standard', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()