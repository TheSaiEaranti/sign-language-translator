"""
tests/benchmark.py

Latency + throughput benchmarks for both models.
Run: python -m tests.benchmark
"""

from __future__ import annotations

import time
import statistics

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from models.architectures import StaticGestureNet, DynamicGestureNet


console = Console()
N_RUNS = 100
WARMUP = 10


def bench_model(model, dummy_input, name: str) -> dict:
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(WARMUP):
            _ = model(dummy_input)

        # Measure
        latencies = []
        for _ in range(N_RUNS):
            start = time.monotonic()
            _ = model(dummy_input)
            latencies.append((time.monotonic() - start) * 1000)

    return {
        "name": name,
        "params": sum(p.numel() for p in model.parameters()),
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(N_RUNS * 0.95)], 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "fps": round(1000 / statistics.median(latencies), 1),
    }


def main():
    console.print("[bold]Model latency benchmarks (CPU)[/bold]\n")

    # Static — single frame
    static_model = StaticGestureNet(num_classes=26)
    static_dummy = torch.randn(1, 2, 21, 3)
    static_result = bench_model(static_model, static_dummy, "StaticGestureNet (letters)")

    # Dynamic — 30-frame sequence
    dynamic_model = DynamicGestureNet(num_classes=100)
    dynamic_dummy = torch.randn(1, 30, 2, 21, 3)
    dynamic_result = bench_model(dynamic_model, dynamic_dummy, "DynamicGestureNet (words)")

    # Batched dynamic
    dynamic_dummy_b = torch.randn(8, 30, 2, 21, 3)
    dynamic_batch_result = bench_model(dynamic_model, dynamic_dummy_b, "DynamicGestureNet (batch=8)")

    table = Table(show_header=True, header_style="bold cyan")
    for col in ["Model", "Params", "p50 (ms)", "p95 (ms)", "Min (ms)", "Max (ms)", "FPS (p50)"]:
        table.add_column(col, justify="right")

    for r in [static_result, dynamic_result, dynamic_batch_result]:
        table.add_row(
            r["name"],
            f"{r['params']:,}",
            str(r["p50_ms"]),
            str(r["p95_ms"]),
            str(r["min_ms"]),
            str(r["max_ms"]),
            str(r["fps"]),
        )

    console.print(table)


if __name__ == "__main__":
    main()
