"""Helpers for the snippet catalog."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class Snippet:
    prefix: str
    title: str
    description: str
    body: str
    categories: tuple[str, ...]
    implementation: str
    extra_aliases: tuple[str, ...] = ()


def snippet(
    prefix: str,
    title: str,
    description: str,
    body: str,
    categories: list[str],
    implementation: str,
    extra_aliases: list[str] | None = None,
) -> Snippet:
    cleaned = textwrap.dedent(body).strip() + "\n"
    return Snippet(prefix, title, description, cleaned, tuple(categories), implementation, tuple(extra_aliases or ()))
