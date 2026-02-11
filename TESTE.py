# test_yolo_simple.py
import cv2
import numpy as np
from ultralytics import YOLO

# ========== CONFIGURAÇÕES (ALTERE AQUI) ==========
VIDEO_PATH = "G:/Petrobras/Sensimar/B062/L1/20210919171301591@DVR-2_Ch3.M4V"
MODEL_PATH = "E:/Raphaela/iSEA/models/corais.pt"
# =================================================

# Carrega modelo e vídeo
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Erro: Não conseguiu abrir o vídeo!")
    exit()

# Configura gravação do resultado
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("resultado_yolo.mp4", fourcc, 
                      cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

print("▶️  Processando... (pressione 'q' para parar e fechar a janela)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ⭐ Corrige problema de memória contígua
    frame_contiguous = np.ascontiguousarray(frame)
    
    # Roda YOLO tracking
    results = model.track(frame_contiguous, persist=True, verbose=False)
    
    # Mostra quantidade de detecções
    num_objs = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"Frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}: {num_objs} objetos")
    
    # Desenha resultados
    frame_plot = results[0].plot() if num_objs > 0 else frame_contiguous

    # ⭐ Mostra preview ao vivo
    cv2.imshow('YOLO Tracking Test', frame_plot)
    
    # Grava no arquivo
    out.write(frame_plot)
    
    # Sai ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ Concluído! Vídeo salvo como: resultado_yolo.mp4")