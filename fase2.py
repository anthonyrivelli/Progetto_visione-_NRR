import cv2
from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_path, conf_threshold):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame, target_classes=None):
        # 1. Usiamo .track() al posto della semplice chiamata al modello
        # 2. Aggiungiamo persist=True per mantenere la memoria degli ID tra un frame e l'altro
        results = self.model.track(frame, conf=self.conf_threshold, persist=True, verbose=False, tracker="bytetrack.yaml", device="cpu")[0] #cuda per usare GPU, cpu per usare cpu
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
            raise ValueError("Impossibile aprire il flusso video")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            # Genera la lista di tutti gli ID direttamente dalle classi del modello
            all_classes = list(self.model.names.keys())  # Genera [0, 1, 2, 3, ..., 79] 
            detections = self.detect(frame, target_classes=all_classes)


            # Classi di test (es. COCO: 0=persona, 1=bici, 2=auto, 3=moto, 5=bus, 7=camion)
            #detections = self.detect(frame, target_classes=[0, 1, 2, 3, 5, 7])

            # Generiamo una lista di 100 colori casuali (formato BGR)
            # Impostiamo il seed per avere sempre gli stessi colori ad ogni avvio
            np.random.seed(100) 
            colors = np.random.randint(0, 255, size=(100, 3), dtype="uint8").tolist()
            
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                class_id = int(det['class_id'])
                # Selezioniamo il colore basandoci sul class_id. 
                # Usiamo il modulo (%) per evitare errori se i class_id superano i 100
                color = tuple(colors[class_id % len(colors)])
                # Disegniamo il rettangolo con il colore della classe
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Mostriamo l'ID insieme a classe e confidenza
                if det['track_id'] is not None:
                    label = f"ID:{det['track_id']} Cls:{det['class_id']} Conf:{det['confidence']:.2f}"
                else:
                    label = f"Cls:{det['class_id']} Conf:{det['confidence']:.2f}"
                
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imshow("Detection & Tracking in tempo reale", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetector(model_path="yolo26n.pt", conf_threshold=0.50)
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

#prova antonella