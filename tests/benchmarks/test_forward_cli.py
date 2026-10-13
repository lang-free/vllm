# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from vllm.benchmarks.forward.batch import build_batch
from vllm.benchmarks.forward.shapes import parse_case

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


def _group(block_size: int, replay: int = 0):
    return SimpleNamespace(
        kv_cache_spec=SimpleNamespace(
            block_size=block_size, prefix_replay_tokens=replay
        )
    )


def test_build_batch_layout():
    """The runner sees each request kind as the scheduler would schedule it."""
    case = parse_case("c=2*1@100,4@60:3,32@0,8@40,16@300h")
    groups = [_group(16), _group(32, replay=128)]
    batch = build_batch(case, groups, 1000, 1000, np.random.default_rng(0))

    # (scheduled, computed, is_decode) per request: decode, verify, prefill,
    # chunked prefill, then a prefix hit replaying the last 128 hit tokens.
    assert list(batch.expected.values()) == [
        (1, 100, True),
        (1, 100, True),
        (4, 60, True),
        (32, 0, False),
        (8, 40, False),
        (16 + 128, 300 - 128, False),
    ]
    output = batch.scheduler_output
    assert output.total_num_scheduled_tokens == 1 + 1 + 4 + 32 + 8 + 144
    assert len(output.scheduled_spec_decode_tokens["bench-2"]) == 3
    assert output.scheduled_new_reqs[-1].replay_start == 300 - 128
    for req_id, (scheduled, _, _) in batch.expected.items():
        assert len(batch.query_token_ids[req_id]) == scheduled
    # Groups share one block pool: block IDs must not overlap.
    ranges = sorted(r for group in batch.group_block_ranges for r in group)
    assert ranges[0][0] == 1
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))

    with pytest.raises(ValueError, match="KV blocks"):
        build_batch(case, groups, 10, 1000, np.random.default_rng(0))


@pytest.mark.benchmark
def test_bench_forward(tmp_path):
    output = tmp_path / "forward.json"
    command = [
        "vllm",
        "bench",
        "forward",
        "--model",
        MODEL_NAME,
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.5",
        "--case",
        "decode=8*1@512",
        "--case",
        "prefill=256@0",
        "--case",
        "mix=128@256,4*1@512",
        "--iters",
        "3",
        "--samples",
        "2",
        "--output-json",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

    results = {row["case"]: row for row in json.loads(output.read_text())}
    assert list(results) == ["decode", "prefill", "mix"]
    for row in results.values():
        rank0 = row["ranks"][0]
        assert len(rank0["batch_order"]) == len(row["requests"])
        assert rank0["finite"] and rank0["graph_deterministic"]
        assert row["median_us"] > 0
