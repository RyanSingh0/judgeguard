from __future__ import annotations

import pytest

from judgeguard.data.load import load_items
from judgeguard.data.schema import Item


@pytest.fixture(scope="session")
def items() -> list[Item]:
    return load_items(24, seed=7)


@pytest.fixture(scope="session")
def item(items: list[Item]) -> Item:
    return items[0]
