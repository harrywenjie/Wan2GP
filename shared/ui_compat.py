"""CLI-friendly stand-ins for the few Gradio component return types still emitted by legacy helpers.

These lightweight dataclasses let the headless runtime describe UI intent without importing Gradio.
Downstream consumers (CLI, tests) can switch on `kind` to interpret responses deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class UIComponent:
    """Minimal representation of a UI response emitted by legacy queue helpers."""

    kind: str
    props: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.props.get(key, default)


def component(kind: str, **props: Any) -> UIComponent:
    return UIComponent(kind=kind, props=props)


def update(**props: Any) -> UIComponent:
    return component("update", **props)


def tabs(**props: Any) -> UIComponent:
    return component("tabs", **props)


def html(**props: Any) -> UIComponent:
    return component("html", **props)


def button(**props: Any) -> UIComponent:
    return component("button", **props)


def column(**props: Any) -> UIComponent:
    return component("column", **props)


def row(**props: Any) -> UIComponent:
    return component("row", **props)


def dropdown(**props: Any) -> UIComponent:
    return component("dropdown", **props)


def textbox(**props: Any) -> UIComponent:
    return component("textbox", **props)


def checkbox(**props: Any) -> UIComponent:
    return component("checkbox", **props)


def text(**props: Any) -> UIComponent:
    return component("text", **props)


def accordion(**props: Any) -> UIComponent:
    return component("accordion", **props)


@dataclass(frozen=True)
class UIEvent:
    """Minimal event payload used to mirror legacy Gradio callbacks."""

    target: Any = None
    value: Any = None
    index: Any = None
    data: Any = None
