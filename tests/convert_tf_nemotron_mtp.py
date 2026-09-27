#!/usr/bin/env python3
"""Convert TensorFold's Nemotron `mtp-4bit.safetensors` into mlx-serve's bare-key
`mtp_head.safetensors` (sevren-ai layout). Pure key rename — tensor bytes and
quant metadata untouched. Idempotent: refuses to overwrite an existing output.

  python3 tests/convert_tf_nemotron_mtp.py <model_dir>
"""
import json
import sys
from pathlib import Path

import mlx.core as mx

RULES = [
    ("layers.0.enorm.", "enorm."),
    ("layers.0.hnorm.", "hnorm."),
    ("layers.0.eh_proj.", "eh_proj."),
    ("layers.0.norm.", "attn.norm."),
    ("layers.0.mixer.", "attn.mixer."),
    ("layers.1.norm.", "moe.norm."),
    ("layers.1.mixer.", "moe.mixer."),
    ("layers.1.final_layernorm.", "final_norm."),
]


def map_key(key: str) -> str | None:
    for src, dst in RULES:
        if key.startswith(src):
            return dst + key[len(src):]
    return None


def main() -> None:
    model_dir = Path(sys.argv[1]).expanduser()
    src = model_dir / "mtp-4bit.safetensors"
    dst = model_dir / "mtp_head.safetensors"
    if not src.is_file():
        sys.exit(f"missing {src}")
    if dst.exists():
        sys.exit(f"refusing to overwrite {dst}")
    weights, meta = mx.load(str(src), return_metadata=True)
    out: dict[str, mx.array] = {}
    for key, value in weights.items():
        mapped = map_key(key)
        if mapped is None:
            sys.exit(f"unmapped key {key!r} — extend RULES")
        out[mapped] = value
    must = ["eh_proj.weight", "attn.mixer.q_proj.weight", "moe.mixer.switch_mlp.fc1.weight",
            "moe.mixer.gate.weight", "final_norm.weight"]
    for name in must:
        if not any(k == name for k in out):
            sys.exit(f"converted file is missing {name}")
    mx.save_safetensors(str(dst), out, metadata=dict(meta))
    print(f"wrote {dst} ({len(out)} tensors) from {src.name} metadata={meta}")


if __name__ == "__main__":
    main()
