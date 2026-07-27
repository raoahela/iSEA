from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition, QMutexLocker
from ultralytics import YOLO
import numpy as np
import torch
import cv2

class DetectionThread(QThread):
    detection_finished = pyqtSignal(object, object, int)
    progress_updated = pyqtSignal(int, int)  # frames_processed, total_frames
    
    def __init__(self, model: YOLO, parent=None):
        super().__init__(parent)
        self.model = model
        self.current_frames = []  # Batch de frames
        self.current_frame_nums = []
        self.running = True
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.cuda_available = torch.cuda.is_available()
        
        # Otimizações para YOLO26
        self.half_precision = True  # FP16 nativo
        self.batch_size = 4  # Processamento em lote
        self.use_fp16 = True  # YOLO26 suporta FP16
        self.max_detections = 100  # Limite para performance
        self.confidence_threshold = 0.55
        self.iou_threshold = 0.65
        
        # Cache de frames para evitar processamento repetido
        self.frame_hash_cache = {}
        self.cache_size = 100
        
    def set_frame(self, frame: np.ndarray, frame_num: int):
        """Adiciona frame ao buffer para processamento em lote"""
        with QMutexLocker(self.mutex):
            # Verifica se frame já foi processado recentemente (deduplicação)
            frame_hash = self._compute_frame_hash(frame)
            if frame_num in self.frame_hash_cache:
                if self.frame_hash_cache[frame_num] == frame_hash:
                    return  # Frame idêntico já processado
                    
            self.frame_hash_cache[frame_num] = frame_hash
            if len(self.frame_hash_cache) > self.cache_size:
                # Remove o mais antigo
                oldest = min(self.frame_hash_cache.keys())
                del self.frame_hash_cache[oldest]
            
            self.current_frames.append(frame)
            self.current_frame_nums.append(frame_num)
            
            # Processa em lote quando atinge o tamanho
            if len(self.current_frames) >= self.batch_size:
                frames_batch = self.current_frames.copy()
                nums_batch = self.current_frame_nums.copy()
                self.current_frames.clear()
                self.current_frame_nums.clear()
                self.condition.wakeOne()
                
    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """Computa hash do frame para deduplicação"""
        small = cv2.resize(frame, (64, 64))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        import hashlib
        return hashlib.blake2b(gray.tobytes(), digest_size=8).hexdigest()

    def set_model(self, model: YOLO | None):
        with QMutexLocker(self.mutex):
            self.model = model

    def stop(self):
        with QMutexLocker(self.mutex):
            self.running = False
            self.condition.wakeOne()
        self.wait()

    def run(self):
        while True:
            with QMutexLocker(self.mutex):
                while not self.current_frames and self.running:
                    self.condition.wait(self.mutex)
                
                if not self.running:
                    break
                    
                frames = self.current_frames.copy()
                frame_nums = self.current_frame_nums.copy()
                self.current_frames.clear()
                self.current_frame_nums.clear()
            
            model = self.model
            if model is None:
                continue

            try:
                # Configuração otimizada para YOLO26
                device = 'cuda' if self.cuda_available else 'cpu'
                
                # YOLO26 com batch inference
                with torch.no_grad():
                    results = model.track(
                        frames,  # Batch de frames
                        persist=True, 
                        verbose=False, 
                        device=device,
                        tracker="botsort.yaml",
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        half=self.half_precision,  # FP16 para velocidade
                        imgsz=640,  # Tamanho otimizado
                        max_det=self.max_detections,
                        vid_stride=False,  # YOLO26 otimizado
                    )
                    
                    # Processa cada resultado
                    for i, result in enumerate(results):
                        if result and len(result) > 0:
                            self.detection_finished.emit(
                                result, 
                                frames[i], 
                                frame_nums[i]
                            )
                            
                # Atualiza progresso
                self.progress_updated.emit(len(frames), len(frames))
                        
            except Exception as e:
                print(f"DetectionThread error: {e}")
                import traceback
                traceback.print_exc()