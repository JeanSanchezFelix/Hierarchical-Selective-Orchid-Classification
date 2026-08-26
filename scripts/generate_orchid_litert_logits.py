#!/usr/bin/env python3
"""Generate identically ordered PyTorch and LiteRT logits for export parity checks."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_compression.src.orchid.edge_audit import load_auditable_model


def primary_output(value: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return value[0] if isinstance(value, tuple) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--litert-model", required=True)
    parser.add_argument("--torch-logits", required=True)
    parser.add_argument("--litert-logits", required=True)
    parser.add_argument("--examples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.examples < 1:
        raise ValueError("--examples must be at least 1.")

    import tensorflow as tf

    model, metadata = load_auditable_model(args.checkpoint)
    image_size = int(metadata["img_size"])
    inputs = np.random.default_rng(args.seed).standard_normal(
        (args.examples, 3, image_size, image_size), dtype=np.float32
    )
    with torch.no_grad():
        torch_outputs = primary_output(model(torch.from_numpy(inputs))).cpu().numpy()

    interpreter = tf.lite.Interpreter(model_path=args.litert_model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    litert_outputs = []
    for sample in inputs:
        interpreter.set_tensor(input_detail["index"], sample[None, ...].astype(input_detail["dtype"], copy=False))
        interpreter.invoke()
        outputs = [interpreter.get_tensor(detail["index"]) for detail in output_details]
        matching = [output for output in outputs if output.ndim == 2 and output.shape[1] == torch_outputs.shape[1]]
        if len(matching) != 1:
            raise RuntimeError(
                "Could not identify the LiteRT species-logit output matching PyTorch shape "
                f"{torch_outputs.shape}; LiteRT outputs were {[output.shape for output in outputs]}."
            )
        litert_outputs.append(matching[0][0])

    for destination, values in ((args.torch_logits, torch_outputs), (args.litert_logits, np.asarray(litert_outputs))):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values)
    print(f"wrote {args.examples} paired synthetic inputs and logits")


if __name__ == "__main__":
    main()
