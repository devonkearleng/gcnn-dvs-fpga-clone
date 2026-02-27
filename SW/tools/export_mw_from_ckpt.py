import argparse
from pathlib import Path
from typing import Dict, Optional

import torch


def load_checkpoint_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    return state_dict


def strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state_dict.items():
        normalized = key
        if normalized.startswith("module."):
            normalized = normalized[len("module."):]
        if normalized.startswith("model."):
            normalized = normalized[len("model."):]
        stripped[normalized] = value
    return stripped


def find_linear_weight_key(state_dict: Dict[str, torch.Tensor], user_key: Optional[str]) -> str:
    if user_key is not None:
        if user_key in state_dict:
            return user_key
        raise KeyError(f"Requested key '{user_key}' not found in checkpoint.")

    preferred = [
        "linear.linear.weight",
        "linear.weight",
    ]

    for key in preferred:
        if key in state_dict:
            return key

    candidates = [
        key
        for key in state_dict.keys()
        if key.endswith("linear.linear.weight") or key.endswith("linear.weight")
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise KeyError(
            "Multiple possible linear weight keys found: "
            + ", ".join(candidates)
            + ". Pass --weight-key explicitly."
        )

    raise KeyError("Could not find linear classifier weight key in checkpoint.")


def find_linear_bias_key(state_dict: Dict[str, torch.Tensor], weight_key: str, user_bias_key: Optional[str]) -> Optional[str]:
    if user_bias_key is not None:
        if user_bias_key in state_dict:
            return user_bias_key
        raise KeyError(f"Requested bias key '{user_bias_key}' not found in checkpoint.")

    if weight_key.endswith(".weight"):
        candidate = weight_key[:-len(".weight")] + ".bias"
        if candidate in state_dict:
            return candidate

    return None


def export_mw(
    ckpt_path: Path,
    out_path: Path,
    weight_key: Optional[str],
    bias_out_path: Optional[Path],
    bias_key: Optional[str],
    transpose: bool,
    expected_num_classes: Optional[int],
) -> None:
    state_dict = load_checkpoint_state_dict(ckpt_path)
    state_dict = strip_prefixes(state_dict)

    key = find_linear_weight_key(state_dict, weight_key)
    weight = state_dict[key].detach().cpu()
    resolved_bias_key = find_linear_bias_key(state_dict, key, bias_key)

    if weight.ndim != 2:
        raise ValueError(f"Expected 2D linear weight tensor, got shape {tuple(weight.shape)}")

    matrix = weight.t() if transpose else weight
    matrix_int = torch.round(matrix).to(torch.int64)

    if expected_num_classes is not None and matrix_int.shape[1] != expected_num_classes:
        raise ValueError(
            f"Exported matrix has {matrix_int.shape[1]} columns, expected {expected_num_classes}. "
            "Check dataset/model or pass the correct --num-classes."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in matrix_int.tolist():
            f.write(" ".join(str(v) for v in row))
            f.write("\n")

    if bias_out_path is not None:
        bias_out_path.parent.mkdir(parents=True, exist_ok=True)
        if resolved_bias_key is None:
            with bias_out_path.open("w", encoding="utf-8") as f:
                f.write(" ".join("0" for _ in range(matrix_int.shape[1])))
                f.write("\n")
            print("Bias key: <not found>; exported zero bias vector")
        else:
            bias = state_dict[resolved_bias_key].detach().cpu()
            if bias.ndim != 1:
                raise ValueError(f"Expected 1D bias tensor, got shape {tuple(bias.shape)}")
            if bias.shape[0] != matrix_int.shape[1]:
                raise ValueError(
                    f"Bias length {bias.shape[0]} does not match number of classes {matrix_int.shape[1]}"
                )
            bias_int = torch.round(bias).to(torch.int64)
            with bias_out_path.open("w", encoding="utf-8") as f:
                f.write(" ".join(str(v) for v in bias_int.tolist()))
                f.write("\n")
            print(f"Bias key: {resolved_bias_key}")
            print(f"Exported mb shape: {bias_int.shape[0]}")
            print(f"Bias output: {bias_out_path}")

    print(f"Checkpoint: {ckpt_path}")
    print(f"Weight key: {key}")
    print(f"Exported mw shape: {matrix_int.shape[0]} x {matrix_int.shape[1]} (rows x cols)")
    print(f"Output: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PS head weights to demo mw.txt format.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to checkpoint file (.ckpt).")
    parser.add_argument("--out", type=Path, default=Path("mw.txt"), help="Output path for mw.txt.")
    parser.add_argument(
        "--weight-key",
        type=str,
        default=None,
        help="Explicit checkpoint key for linear weights (optional).",
    )
    parser.add_argument(
        "--bias-key",
        type=str,
        default=None,
        help="Explicit checkpoint key for linear bias (optional).",
    )
    parser.add_argument(
        "--bias-out",
        type=Path,
        default=None,
        help="Optional output path for mb.txt (classifier bias vector).",
    )
    parser.add_argument(
        "--no-transpose",
        action="store_true",
        help="Do not transpose the weight matrix before export.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Validate exported matrix column count equals this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_mw(
        ckpt_path=args.ckpt,
        out_path=args.out,
        weight_key=args.weight_key,
        bias_out_path=args.bias_out,
        bias_key=args.bias_key,
        transpose=not args.no_transpose,
        expected_num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
