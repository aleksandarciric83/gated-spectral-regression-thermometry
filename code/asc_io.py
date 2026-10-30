"""Reader for Andor Solis .asc kinetic-series exports (one file per temperature).
parse_asc and parse_temperature are taken unchanged from analyze_spectra.py of the measurement folder."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def parse_temperature(filename: str) -> int | None:
    match = re.match(r"^(\d+)C", filename)
    return int(match.group(1)) if match else None


def parse_asc(path: Path) -> dict:
    lines = path.read_text().splitlines()
    header: dict[str, str] = {}
    data_start = 0
    for i, line in enumerate(lines):
        if "\t" in line and not line.startswith("Date") and ":" not in line.split("\t")[0]:
            try:
                float(line.split("\t")[0])
                data_start = i
                break
            except ValueError:
                pass
        if ":" in line and "\t" not in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()

    if "Number in Kinetics Series" not in header:
        wavelengths: list[float] = []
        counts: list[float] = []
        for line in lines[data_start:]:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            try:
                wavelengths.append(float(parts[0]))
                counts.append(float(parts[1]))
            except ValueError:
                continue
        wl_arr = np.asarray(wavelengths)
        data = np.asarray(counts, dtype=float)[:, np.newaxis]
        return {
            "path": path,
            "header": header,
            "wavelength_nm": wl_arr,
            "delays_us": np.array([0.0]),
            "counts": data,
            "rate": data,
            "temperature_c": parse_temperature(path.name),
        }

    n_kin = int(header["Number in Kinetics Series"])
    gate0 = float(header["Gate Delay (nsecs)"])
    gate_step = float(header["Gate Delay Step (nsecs)"])
    exp_time = float(header["Exposure Time (secs)"])
    n_acc = int(header["Number of Accumulations"])

    wavelengths: list[float] = []
    counts: list[list[float]] = []
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        try:
            wl = float(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if len(vals) != n_kin:
            continue
        wavelengths.append(wl)
        counts.append(vals)

    wl_arr = np.asarray(wavelengths)
    data = np.asarray(counts, dtype=float)
    delays_ns = gate0 + np.arange(n_kin) * gate_step
    # Normalize to counts per second per accumulation.
    rate = data / (exp_time * n_acc)

    return {
        "path": path,
        "header": header,
        "wavelength_nm": wl_arr,
        "delays_us": delays_ns / 1000.0,
        "counts": data,
        "rate": rate,
        "temperature_c": parse_temperature(path.name),
    }
