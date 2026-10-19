import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path, conf_threshold):
        """
        Inizializza il rilevatore caricando il modello Deep Learning.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame, target_classes=None):
        """
        Analizza un singolo frame per rilevare gli oggetti. Restituisce una lista
        di dizionari contenenti bounding box, confidenza e ID della classe.
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            # Filtra i risultati mantenendo solo le classi di interesse se specificate
            if target_classes is None or class_id in target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id
                })
                
        return detections

    def process_video_stream(self, video_path):
        """
        Elabora il flusso video in tempo reale garantendo un frame rate adeguato.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Impossibile aprire il flusso video.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Classi di test comuni (es. COCO: 0=persona, 2=auto, 3=moto, 5=bus, 7=camion)
            detections = self.detect(frame, target_classes=[0, 1, 2, 3, 5, 7])
            
            # Visualizzazione preliminare delle bounding box sul frame
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"Class: {det['class_id']} Conf: {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(detections) 
            print("\n")
            cv2.imshow("Detection in tempo reale", frame)
            
            
            # Interrompe l'esecuzione premendo 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
               
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Inizializzazione della classe ed esecuzione su un video pre-registrato
    detector = ObjectDetector(model_path="yolo26n.pt", conf_threshold=0.50)
    detector.process_video_stream("video/video_traffico.mp4")