import os
import time
import copy
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Callable, Optional, Any

from src.utils import load_data, process_callbacks
from src.Quantization.quantization_utils.conversions.onnx import export_pytorch_to_onnx, export_onnx_to_savedmodel, export_savedmodel_to_tflite
from src.train.knowledge_distillation import train_qat_kd
from src.utils.logging_setup import configure_logging


def main():
    dataset = "SkinCancer"
    batch_size = 32
    learning_rate = 0.001
    epochs = 1
    save_dir = f"models/{dataset}/Quantized"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = "mobilenet_v2"
    student_model = "mobilenet_v2"
    args = {"callbacks": ["ModelCheckpoint", "EarlyStopping", "ReduceLROnPlateau"], 
            "save_dir": save_dir, 
            "model_name": student_model,
            "save_name": "qat_kd"}
    CALLBACKS = list(process_callbacks(args).values())

    os.makedirs(save_dir, exist_ok=True)

    dataloaders = load_data(dataset=dataset, batch_size=batch_size)

    teacher_model_weights = 'models/SkinCancer/mobilenet_v2_best_model.pth'  # Update with your saved model weights

    criterion = "cross_entropy"
    optimizer = "adam"

    # 1. Train model
    teacher, quantized_student = train_qat_kd(
                                            teacher_name=teacher_model, 
                                            student_name=student_model,
                                            data_loaders=dataloaders,
                                            save_dir=save_dir,
                                            learning_rate=learning_rate,
                                            epochs=epochs,
                                            criterion = criterion,
                                            optimizer = optimizer,
                                            callbacks=CALLBACKS,
                                            quant_mode="export",
                                            teacher_model_weights=teacher_model_weights,
                                            device=device,
                                    )
    
    # 2. Export to ONNX
    example_inputs = next(iter(dataloaders["train"]))[0].to(device)
    onnx_path = os.path.join(save_dir, f"{student_model}_qat_kd.onnx")
    export_pytorch_to_onnx(quantized_student, example_inputs, onnx_path)

    # 3. Convert ONNX → TensorFlow SavedModel
    tf_saved_model_dir = os.path.join(save_dir, f"{student_model}_saved_model")
    export_onnx_to_savedmodel(onnx_path, tf_saved_model_dir)

    # 4. Convert SavedModel → TFLite
    tflite_path = os.path.join(save_dir, f"{student_model}_qat_kd.tflite")
    export_savedmodel_to_tflite(tf_saved_model_dir, tflite_path, dataloaders["train"])

if __name__ == "__main__":
    main()