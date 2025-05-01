import cv2
from ultralytics import YOLO


cap = cv2.VideoCapture(0)


model = YOLO('best2.pt') 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

   
    results = model(frame)[0]

    
    for box in results.boxes:
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f'{model.names[cls]} {conf:.2f}'

        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    
    cv2.imshow('YOLOv8 Detection', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
