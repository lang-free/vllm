# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact batch shapes for `vllm bench forward`."""

import json
from dataclasses import dataclass
from pathlib import Path

import regex as re


@dataclass(frozen=True)
class BenchRequest:
    query_len: int
    # KV length after the step: the prefix already in the KV cache plus the
    # query tokens.
    kv_len: int
    # Spec-decode verify: query_len == 1 + num_draft_tokens.
    num_draft_tokens: int = 0
    # Prefill after a prefix-cache hit of the whole prefix. Models with SWA
    # bounded replay (DeepSeek-V4.1) then recompute the hit's last window.
    prefix_hit: bool = False

    def __post_init__(self) -> None:
        if (
            self.query_len < 1
            or self.kv_len < self.query_len
            or self.num_draft_tokens < 0
        ):
            raise ValueError(f"Invalid request: {self}")
        if self.num_draft_tokens and not self.is_decode:
            raise ValueError(
                f"Verify needs query_len == 1 + num_draft_tokens and a prefix: {self}"
            )
        if self.prefix_hit and (self.is_decode or not self.prefix_len):
            raise ValueError(f"A prefix hit needs a prefill with a prefix: {self}")

    @property
    def prefix_len(self) -> int:
        return self.kv_len - self.query_len

    @property
    def is_decode(self) -> bool:
        """Decode or verify; otherwise a (chunked) prefill."""
        return self.prefix_len > 0 and self.query_len == 1 + self.num_draft_tokens


@dataclass(frozen=True)
class BenchCase:
    name: str
    requests: tuple[BenchRequest, ...]

    @property
    def num_tokens(self) -> int:
        return sum(r.query_len for r in self.requests)

    @property
    def max_kv_len(self) -> int:
        return max(r.kv_len for r in self.requests)


_REQ_RE = re.compile(r"^(?:(\d+)\*)?(\d+)(?:@(\d+))?(h)?(?::(\d+))?$")


def parse_case(text: str) -> BenchCase:
    """Parse `NAME=REQS`, where REQS is a comma list of `[N*]Q[@K][h][:D]`.

    Q is the query length, K the KV length including the query (default Q, a
    fresh prefill), D the number of draft tokens (verify) and N the number of
    such requests; `h` marks a prefill after a prefix hit. For example,
    `decode=32*1@8192`, `verify=16*4@4096:3`, `prefill=8192`,
    `hit=8192@40960h` or `mix=8192,32*1@8192`.
    """
    name, sep, body = text.partition("=")
    if not sep:
        name, body = text, text
    requests: list[BenchRequest] = []
    for item in body.split(","):
        match = _REQ_RE.match(item.strip())
        if match is None:
            raise ValueError(f"Invalid request {item!r} in case {text!r}")
        count, query, kv, hit, drafts = match.groups()
        request = BenchRequest(
            int(query), int(kv or query), int(drafts or 0), prefix_hit=bool(hit)
        )
        requests.extend([request] * int(count or 1))
    return BenchCase(name, tuple(requests))


def load_cases(path: str) -> list[BenchCase]:
    """Load cases from a JSON or YAML file.

    Format: `{"cases": [{"name": ..., "requests": [{"query_len": ...,
    "kv_len": ..., "num_draft_tokens": ..., "count": ...}]}]}`.
    """
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    cases = []
    for case in data["cases"]:
        requests: list[BenchRequest] = []
        for req in case["requests"]:
            req = dict(req)
            count = req.pop("count", 1)
            requests.extend([BenchRequest(**req)] * count)
        cases.append(BenchCase(case["name"], tuple(requests)))
    return cases
