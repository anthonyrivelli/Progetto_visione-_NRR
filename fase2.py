import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path, conf_threshold):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame, target_classes=None):
        # 1. Usiamo .track() al posto della semplice chiamata al modello
        # 2. Aggiungiamo persist=True per mantenere la memoria degli ID tra un frame e l'altro
        results = self.model.track(frame, conf=self.conf_threshold, persist=True, verbose=False)[0]
        detections = []

        # Controlliamo se ci sono box e se il tracker ha assegnato degli ID
        if results.boxes is not None:
            # Estraiamo gli ID se disponibili (potrebbero essere None se il tracker sta inizializzando)
            if results.boxes.id is not None:
                track_ids = results.boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(results.boxes)

            # Iteriamo sui box accoppiandoli con i loro ID
            for box, track_id in zip(results.boxes, track_ids):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Filtra i risultati mantenendo solo le classi di interesse
                if target_classes is None or class_id in target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "track_id": track_id  # Aggiungiamo l'ID di tracciamento
                    })
                
        return detections

    def process_video_stream(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Impossibile aprire il flusso video.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Classi di test (es. COCO: 0=persona, 2=auto, 3=moto, 5=bus, 7=camion)
            detections = self.detect(frame, target_classes=[0, 1, 2, 3, 5, 7])
            
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Mostriamo l'ID insieme a classe e confidenza
                id_text = f"ID:{det['track_id']} " if det['track_id'] is not None else ""
                label = f"{id_text}Cls:{det['class_id']} Conf:{det['confidence']:.2f}"
                
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("Detection & Tracking in tempo reale", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
               
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # N.B. Ho ipotizzato yolo11n.pt o yolov8n.pt (yolo26n credo fosse un typo)
    detector = ObjectDetector(model_path="yolov8n.pt", conf_threshold=0.50)
    detector.process_video_stream("video/video_traffico.mp4")

# VESIONE NATIVA DI ULTRALYTICS YOLO
# from ultralytics import YOLO

# # Carica il modello
# model = YOLO("yolov8n.pt") 

# # Esegui il tracker. 
# # "classes=[0, 1, 2, 3, 5, 7]" fa esattamente quello che faceva il tuo "target_classes"
# results = model.track(
#     source="video/video_traffico.mp4", 
#     conf=0.50, 
#     classes=[0, 1, 2, 3, 5, 7], 
#     show=True,  # Mostra la finestra video automaticamente
#     tracker="botsort.yaml" # (Opzionale) puoi usare "bytetrack.yaml" se preferisci un altro algoritmo
# )