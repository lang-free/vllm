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
    # Tokens already in the KV cache.
    context_len: int
    # Spec-decode verify: query_len == 1 + num_draft_tokens.
    num_draft_tokens: int = 0
    # Prefill after a prefix-cache hit of context_len tokens. Models with SWA
    # bounded replay (DeepSeek-V4.1) then recompute the hit's last window.
    prefix_hit: bool = False

    def __post_init__(self) -> None:
        if self.query_len < 1 or self.context_len < 0 or self.num_draft_tokens < 0:
            raise ValueError(f"Invalid request: {self}")
        if self.num_draft_tokens and not self.is_decode:
            raise ValueError(
                f"Verify needs query_len == 1 + num_draft_tokens and context: {self}"
            )
        if self.prefix_hit and (self.is_decode or not self.context_len):
            raise ValueError(f"A prefix hit needs a prefill with context: {self}")

    @property
    def is_decode(self) -> bool:
        """Decode or verify; otherwise a (chunked) prefill."""
        return self.context_len > 0 and self.query_len == 1 + self.num_draft_tokens

    @property
    def seq_len(self) -> int:
        return self.context_len + self.query_len


@dataclass(frozen=True)
class BenchCase:
    name: str
    requests: tuple[BenchRequest, ...]

    @property
    def num_tokens(self) -> int:
        return sum(r.query_len for r in self.requests)

    @property
    def max_seq_len(self) -> int:
        return max(r.seq_len for r in self.requests)


_REQ_RE = re.compile(r"^(?:(\d+)\*)?(\d+)@(\d+)(h)?(?::(\d+))?$")


def parse_case(text: str) -> BenchCase:
    """Parse `NAME=REQS`, where REQS is a comma list of `[N*]Q@C[h][:D]`.

    Q is the query length, C the context length, D the number of draft tokens
    (verify) and N the number of such requests; `h` marks a prefill after a
    prefix hit. For example, `decode=32*1@8192`, `verify=16*4@4096:3`,
    `hit=8192@32768h` or `mix=8192@0,32*1@8192`.
    """
    name, sep, body = text.partition("=")
    if not sep:
        name, body = text, text
    requests: list[BenchRequest] = []
    for item in body.split(","):
        match = _REQ_RE.match(item.strip())
        if match is None:
            raise ValueError(f"Invalid request {item!r} in case {text!r}")
        count, query, context, hit, drafts = match.groups()
        request = BenchRequest(
            int(query), int(context), int(drafts or 0), prefix_hit=bool(hit)
        )
        requests.extend([request] * int(count or 1))
    return BenchCase(name, tuple(requests))


def load_cases(path: str) -> list[BenchCase]:
    """Load cases from a JSON or YAML file.

    Format: `{"cases": [{"name": ..., "requests": [{"query_len": ...,
    "context_len": ..., "num_draft_tokens": ..., "count": ...}]}]}`.
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
