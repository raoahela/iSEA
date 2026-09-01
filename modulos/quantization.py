# quantization_fp16_pt.py
import torch
from ultralytics import YOLO

def quantize_fp16_pt(model_path, output_path):
    """
    Converte modelo para FP16 (half precision)
    2x speedup, arquivo menor
    """
    # Carrega modelo
    model = YOLO(model_path)
    
    # Converte para FP16
    model.model = model.model.half()  # Converte todos os parâmetros para float16
    
    # Salva
    torch.save({
        'model': model.model,
        'names': model.names,
        'task': model.task,
        'version': 'fp16'
    }, output_path)
    
    print(f"Modelo FP16 salvo: {output_path}")
    return output_path

# Uso mais simples possível
quantize_fp16_pt(
    r"E:\Raphaela\iSEA\modelos\corais.pt",
    r"E:\Raphaela\iSEA\modelos\corais_fp16.pt"
)