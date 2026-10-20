import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict

class ObjectDetector:
    def __init__(self, model_path, conf_threshold, min_hits=3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # --- STRUTTURE DATI PER LA FASE 3 (CONTEGGIO) ---
        # Set globale di tutti i track_id unici confermati
        self.counted_ids = set()
        
        # Dizionario che mappa: class_id -> set di track_id unici per quella specifica classe
        self.class_counts = defaultdict(set)
        
        # Contatore di apparizioni per ogni track_id (per filtrare i falsi positivi)
        self.id_frames_seen = defaultdict(int)
        self.min_hits = min_hits  # Numero minimo di frame prima di contare l'oggetto

    def detect(self, frame, target_classes=None):
        results = self.model.track(
            frame, 
            conf=self.conf_threshold, 
            persist=True, 
            verbose=False, 
            tracker="bytetrack.yaml", 
            device="cpu" # Cambia in "cuda" se usi la GPU
        )[0]
        
        detections = []

        if results.boxes is not None:
            if results.boxes.id is not None:
                track_ids = results.boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(results.boxes)

            for box, track_id in zip(results.boxes, track_ids):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if target_classes is None or class_id in target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "track_id": track_id
                    })
                
        return detections

    def update_counts(self, detections):
        """
        Aggiorna i conteggi evitando duplicati grazie ai set sugli ID di tracking.
        """
        current_frame_count = 0

        for det in detections:
            track_id = det["track_id"]
            class_id = det["class_id"]

            # Se il tracker ha assegnato un ID valido
            if track_id is not None:
                current_frame_count += 1
                self.id_frames_seen[track_id] += 1

                # Contiamo l'oggetto solo se è stato visto per almeno 'min_hits' frame
                if self.id_frames_seen[track_id] >= self.min_hits:
                    self.counted_ids.add(track_id)
                    self.class_counts[class_id].add(track_id)

        return current_frame_count

    def draw_counters(self, frame, current_frame_count):
        """
        Disegna un pannello riepilogativo sul frame con il conteggio in tempo reale.
        """
        # Creiamo uno sfondo semitrasparente in alto a sinistra per rendere leggibile il testo
        overlay = frame.copy()
        box_height = 70 + (len(self.class_counts) * 25)
        cv2.rectangle(overlay, (10, 10), (320, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Conteggio totale cumulativo e presenti nel frame attuale
        total_unique = len(self.counted_ids)
        cv2.putText(frame, f"Totale unici: {total_unique}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Attuali nel frame: {current_frame_count}", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Dettaglio conteggio per singola classe
        y_offset = 90
        for class_id, ids_set in self.class_counts.items():
            class_name = self.model.names[class_id]
            count = len(ids_set)
            cv2.putText(frame, f"- {class_name}: {count}", (25, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 25

    def process_video_stream(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Impossibile aprire il flusso video")

        # Inizializziamo classi e colori FUORI dal ciclo while per ottimizzare gli FPS
        # Classi di interesse (es. COCO: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck)
        target_classes = [0, 1, 2, 3, 5, 7]
        
        np.random.seed(100)
        colors = np.random.randint(0, 255, size=(100, 3), dtype="uint8").tolist()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Rilevamento e Tracking (Fasi 1 e 2)
            detections = self.detect(frame, target_classes=target_classes)

            # 2. Aggiornamento Conteggi (Fase 3)
            current_frame_count = self.update_counts(detections)

            # 3. Disegno Bounding Box e ID
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                class_id = int(det["class_id"])
                class_name = self.model.names[class_id]
                color = tuple(colors[class_id % len(colors)])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                if det["track_id"] is not None:
                    label = f"ID:{det['track_id']} {class_name} {det['confidence']:.2f}"
                else:
                    label = f"{class_name} {det['confidence']:.2f}"
                
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 4. Disegno del pannello con i conteggi aggiornati
            self.draw_counters(frame, current_frame_count)

            cv2.imshow("Detection, Tracking & Counting in tempo reale", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectDetector(model_path="yolo26n.pt", conf_threshold=0.50, min_hits=3)
    detector.process_video_stream("video/video_traffico.mp4")