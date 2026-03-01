import argparse
import os
import glob
import re
from typing import Iterable, List, Tuple

import numpy as np


EVT_DVS = 0


def read_bits(arr, mask=None, shift=None):
    if mask is not None:
        arr = arr & mask
    if shift is not None:
        arr = arr >> shift
    return arr


def skip_header(fp):
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "#":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    return p


def load_raw_events(fp, bytes_skip=0, bytes_trim=0, filter_dvs=False, times_first=False):
    valid_mask = 0x80000000
    valid_shift = 31

    p = skip_header(fp)
    fp.seek(p + bytes_skip)
    data = fp.read()
    if bytes_trim > 0:
        data = data[:-bytes_trim]

    data = np.fromstring(data, dtype=">u4")
    if len(data) % 2 != 0:
        raise ValueError("odd number of data elements")

    raw_addr = data[::2]
    timestamp = data[1::2]

    if times_first:
        timestamp, raw_addr = raw_addr, timestamp

    if filter_dvs:
        valid = read_bits(raw_addr, valid_mask, valid_shift) == EVT_DVS
        timestamp = timestamp[valid]
        raw_addr = raw_addr[valid]

    return timestamp, raw_addr


def parse_raw_address(addr, x_mask, x_shift, y_mask, y_shift, polarity_mask, polarity_shift):
    polarity = read_bits(addr, polarity_mask, polarity_shift).astype(np.bool_)
    x = read_bits(addr, x_mask, x_shift)
    y = read_bits(addr, y_mask, y_shift)
    return x, y, polarity


def load_events(fp, x_mask, x_shift, y_mask, y_shift, polarity_mask, polarity_shift):
    timestamp, addr = load_raw_events(fp, filter_dvs=False)
    x, y, polarity = parse_raw_address(
        addr,
        x_mask=x_mask,
        x_shift=x_shift,
        y_mask=y_mask,
        y_shift=y_shift,
        polarity_mask=polarity_mask,
        polarity_shift=polarity_shift,
    )
    return timestamp, x, y, polarity


def iter_mnistdvs_files_standard(data_dir: str, split: str) -> List[Tuple[str, int]]:
    pattern = os.path.join(data_dir, "mnist-dvs", split, "*", "mnist_*.aedat")
    files = sorted(glob.glob(pattern))
    items = []
    for path in files:
        label = int(os.path.basename(os.path.dirname(path)))
        items.append((path, label))
    return items


def iter_mnistdvs_files_grabbed(data_dir: str, mnist_scale: str) -> List[Tuple[str, int]]:
    if mnist_scale == "all":
        pattern = os.path.join(data_dir, "data*", "grabbed_data*", "scale*", "mnist_*.aedat")
    else:
        pattern = os.path.join(data_dir, "data*", "grabbed_data*", f"scale{mnist_scale}", "mnist_*.aedat")

    files = sorted(glob.glob(pattern))
    items = []
    for path in files:
        label_match = re.search(r"/data(\d+)/", path)
        if label_match is None:
            continue
        label = int(label_match.group(1))
        items.append((path, label))
    return items


def iter_cifar10_files(data_dir: str, split: str) -> List[Tuple[str, int]]:
    class_dict = {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,
    }
    pattern = os.path.join(data_dir, "cifar10-dvs", split, "*", "cifar10_*.aedat")
    files = sorted(glob.glob(pattern))
    items = []
    for path in files:
        class_name = os.path.basename(os.path.dirname(path))
        if class_name not in class_dict:
            continue
        items.append((path, class_dict[class_name]))
    return items


def iter_ncars_files(data_dir: str, split: str) -> List[Tuple[str, int]]:
    items = []

    pattern = os.path.join(data_dir, "ncars", split, "*", "events.txt")
    files = sorted(glob.glob(pattern))
    for path in files:
        label_path = path.replace("events.txt", "is_car.txt")
        if not os.path.exists(label_path):
            continue
        label = int(np.loadtxt(label_path))
        items.append((path, label))

    if items:
        return items

    split_name = "n-cars_train" if split == "train" else "n-cars_test"
    base_dir = os.path.join(data_dir, "n-cars", "Prophesee_Dataset_n_cars", split_name)
    for class_name, label in (("cars", 1), ("background", 0)):
        dat_pattern = os.path.join(base_dir, class_name, "*_td.dat")
        for path in sorted(glob.glob(dat_pattern)):
            items.append((path, label))

    return items


def load_ncars_events_dat(dat_file: str) -> np.ndarray:
    with open(dat_file, "rb") as f:
        num_comment_lines = 0
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.startswith(b"%"):
                f.seek(pos)
                break
            num_comment_lines += 1

        event_size = 8
        if num_comment_lines > 0:
            _event_type = f.read(1)
            event_size_raw = f.read(1)
            if len(event_size_raw) == 1:
                event_size = int.from_bytes(event_size_raw, byteorder="little", signed=False)

        bof = f.tell()
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        num_events = (file_size - bof) // event_size

        f.seek(bof, os.SEEK_SET)
        ts = np.fromfile(f, dtype="<u4", count=num_events)

        f.seek(bof + 4, os.SEEK_SET)
        addr = np.fromfile(f, dtype="<u4", count=num_events)

    xmask = 0x00003FFF
    ymask = 0x0FFFC000
    polmask = 0x10000000
    yshift = 14
    polshift = 28

    x = (addr & xmask).astype(np.int32)
    y = ((addr & ymask) >> yshift).astype(np.int32)
    p = ((addr & polmask) >> polshift).astype(np.int32)
    t = ts.astype(np.int64)

    return np.stack((x, y, t, p), axis=1)


def write_events_txt(out_path: str, events: np.ndarray) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for x, y, t, p in events:
            f.write(f"{int(x)} {int(y)} {int(t)} {int(p)}\n")


def preprocess_mnistdvs_events(
    events: np.ndarray,
    remove_75hz: bool = True,
    stabilize: bool = True,
    remove_polarity: bool = True,
    noise_factor: float = 2.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Replicate the IMSE-CNM process_mnist.m MATLAB preprocessing in Python.

    events: (N, 4) int32 array with columns [x_out, y_out, t_us, p_out]
        where x_out = 127 - y_raw  (col 0, sent to HW as x)
              y_out = 127 - x_raw  (col 1, sent to HW as y) = x_CIN
              t_us                 (col 2, raw microsecond timestamp)
              p_out = 1 - p_raw   (col 3)

    Coordinate identities (from mat2dat.m inverse):
        x_CIN = 127 - x_raw = y_out (col 1)
        y_CIN = y_raw        = 127 - x_out (127 - col 0)

    Stabilisation in output-space:
        tx_new = tx + yc - 63   (tx = col 0 = 127 - y_CIN)
        ty_new = ty - xc + 63   (ty = col 1 = x_CIN)
    """
    if rng is None:
        rng = np.random.default_rng()

    events = events.copy().astype(np.float64)

    # --- a) Remove 75 Hz LCD harmonic ---
    if remove_75hz and len(events) > 1:
        t = events[:, 2]
        dii = np.diff(t)
        mean_dii = float(np.mean(dii))
        std_dii = float(np.std(dii))
        n = len(t)
        ii2 = np.cumsum(mean_dii + noise_factor * std_dii * rng.standard_normal(n))
        ii2 -= np.min(ii2)
        ii_max = np.max(t)
        ii2_max = np.max(ii2)
        if ii2_max > 0:
            ii2 = np.round(ii2 * ii_max / ii2_max)
        events[:, 2] = np.sort(ii2)

    # --- b) Stabilise digit at centre (63, 63) ---
    if stabilize and len(events) > 0:
        TT = 0.2982
        Trampy = TT * 1e6       # 298 200 µs
        Trampx = 2.0 * TT * 1e6  # 596 400 µs
        DY, DX = 5.0, 10.0
        Ymin, Xmin = 58.0, 58.0

        t = events[:, 2]
        yc0 = np.abs(np.mod(t, 2.0 * Trampy) - Trampy) / Trampy * DY + Ymin
        yc = 127.0 - np.round(yc0)   # centre y in sensor coords
        xc0 = np.abs(np.mod(t + 1.5 * Trampx, 2.0 * Trampx) - Trampx) / Trampx * DX + Xmin
        xc = 127.0 - np.round(xc0)   # centre x in sensor coords

        # Apply shift in output-space coordinates
        tx_new = events[:, 0] + yc - 63.0
        ty_new = events[:, 1] - xc + 63.0

        mask = (tx_new >= 0) & (tx_new <= 127) & (ty_new >= 0) & (ty_new <= 127)
        events[:, 0] = tx_new
        events[:, 1] = ty_new
        events = events[mask]

    # --- c) Remove polarity (set all p_out = 1) ---
    if remove_polarity and len(events) > 0:
        events[:, 3] = 1.0

    return events.astype(np.int32)


def export_mnistdvs(items: Iterable[Tuple[str, int]], out_dir: str, time_window_us: int, max_samples: int, preprocess: bool = False, preprocess_seed: int = None) -> Tuple[List[str], List[int]]:
    output_files = []
    labels = []
    rng = np.random.default_rng(preprocess_seed) if preprocess else None
    for idx, (path, label) in enumerate(items):
        if max_samples is not None and idx >= max_samples:
            break
        with open(path, "rb") as fp:
            t, x, y, p = load_events(
                fp,
                x_mask=0xFE,
                x_shift=1,
                y_mask=0x7F00,
                y_shift=8,
                polarity_mask=1,
                polarity_shift=None,
            )
        events = np.stack((127 - y, 127 - x, t, 1 - p.astype(int)), axis=1)
        if preprocess:
            events = preprocess_mnistdvs_events(events, rng=rng)
        events = events[events[:, 2] < time_window_us]

        out_name = f"s{idx:07d}.txt"
        out_path = os.path.join(out_dir, out_name)
        write_events_txt(out_path, events)
        output_files.append(out_name)
        labels.append(label)
    return output_files, labels


def export_cifar10(items: Iterable[Tuple[str, int]], out_dir: str, time_window_us: int, max_samples: int) -> Tuple[List[str], List[int]]:
    output_files = []
    labels = []
    for idx, (path, label) in enumerate(items):
        if max_samples is not None and idx >= max_samples:
            break
        with open(path, "rb") as fp:
            t, x, y, p = load_events(
                fp,
                x_mask=0xFE,
                x_shift=1,
                y_mask=0x7F00,
                y_shift=8,
                polarity_mask=1,
                polarity_shift=None,
            )
        events = np.stack((127 - y, 127 - x, t, 1 - p.astype(int)), axis=1)
        events = events[events[:, 2] < time_window_us]

        out_name = f"sample_{idx:06d}.txt"
        out_path = os.path.join(out_dir, out_name)
        write_events_txt(out_path, events)
        output_files.append(out_name)
        labels.append(label)
    return output_files, labels


def export_ncars(items: Iterable[Tuple[str, int]], out_dir: str, time_window_s: float, time_scale: float, max_samples: int) -> Tuple[List[str], List[int]]:
    output_files = []
    labels = []
    threshold_us = int(round(time_window_s * time_scale))

    for idx, (path, label) in enumerate(items):
        if max_samples is not None and idx >= max_samples:
            break

        if path.endswith(".dat"):
            events = load_ncars_events_dat(path)
            events = events[events[:, 2] < threshold_us]
        else:
            events = np.loadtxt(path)
            if events.ndim == 1:
                events = events.reshape(1, -1)
            mask = events[:, 2] < time_window_s
            events = events[mask]
            events[:, 2] = np.round(events[:, 2] * time_scale)

        out_name = f"sample_{idx:06d}.txt"
        out_path = os.path.join(out_dir, out_name)
        write_events_txt(out_path, events)
        output_files.append(out_name)
        labels.append(label)
    return output_files, labels


def write_manifest(out_dir: str, files: List[str], labels: List[int]) -> None:
    manifest_path = os.path.join(out_dir, "manifest.txt")
    labels_path = os.path.join(out_dir, "labels.txt")
    with open(manifest_path, "w", encoding="utf-8") as f_manifest, open(labels_path, "w", encoding="utf-8") as f_labels:
        for name, label in zip(files, labels):
            f_manifest.write(f"{name} {label}\n")
            f_labels.write(f"{label}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export raw event datasets to SD-card text format.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnistdvs", "cifar10", "ncars"])
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--out-dir", type=str, default="sd_export")
    parser.add_argument("--time-window-us", type=int, default=200000)
    parser.add_argument("--time-window-s", type=float, default=0.1)
    parser.add_argument("--ncars-time-scale", type=float, default=1e6)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--mnist-layout", type=str, default="standard", choices=["standard", "grabbed"])
    parser.add_argument("--mnist-scale", type=str, default="all", choices=["all", "4", "8", "16"])
    parser.add_argument("--preprocess-mnist", action="store_true",
                        help="Apply IMSE-CNM process_mnist.m preprocessing: "
                             "remove 75 Hz LCD harmonic, stabilise digit at centre, "
                             "remove polarity (all events set to p=1). "
                             "Recommended when exporting grabbed MNIST-DVS data.")
    parser.add_argument("--preprocess-seed", type=int, default=None,
                        help="Random seed for the 75 Hz harmonic removal step (reproducibility).")

    args = parser.parse_args()

    if args.dataset == "mnistdvs" and args.mnist_layout == "grabbed":
        splits = ["all"]
    else:
        splits = [args.split] if args.split != "all" else ["train", "test"]

    for split in splits:
        out_dir = os.path.join(args.out_dir, args.dataset, split)
        os.makedirs(out_dir, exist_ok=True)

        if args.dataset == "mnistdvs":
            if args.mnist_layout == "standard":
                items = iter_mnistdvs_files_standard(args.data_dir, split)
            else:
                items = iter_mnistdvs_files_grabbed(args.data_dir, args.mnist_scale)
            files, labels = export_mnistdvs(items, out_dir, args.time_window_us, args.max_samples,
                                            preprocess=args.preprocess_mnist,
                                            preprocess_seed=args.preprocess_seed)
        elif args.dataset == "cifar10":
            items = iter_cifar10_files(args.data_dir, split)
            files, labels = export_cifar10(items, out_dir, args.time_window_us, args.max_samples)
        else:
            items = iter_ncars_files(args.data_dir, split)
            files, labels = export_ncars(items, out_dir, args.time_window_s, args.ncars_time_scale, args.max_samples)

        write_manifest(out_dir, files, labels)
        print(f"Exported {len(files)} samples to {out_dir}")


if __name__ == "__main__":
    main()
