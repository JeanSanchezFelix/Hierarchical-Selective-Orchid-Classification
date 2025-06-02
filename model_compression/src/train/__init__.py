# model_compression/src/train/__init__.py
"""
Transfer Learning and Knowledge Distillation pipeline
"""
from .train import transfer_learning
from .knowledge_distillation import train_kd

__all__ = [
    'transfer_kearning',
    'train_kd'
]
