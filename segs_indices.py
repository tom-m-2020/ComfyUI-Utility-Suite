from __future__ import annotations


def parse_index_csv(index_text: str, node_name: str) -> list[int]:
    if not isinstance(index_text, str):
        raise TypeError(f"{node_name} manual index must be a string.")
    parts = index_text.split(",")
    if any(not part.strip() for part in parts):
        raise ValueError(f"{node_name} manual index contains an empty element.")
    try:
        return [int(part.strip()) for part in parts]
    except ValueError as error:
        raise ValueError(f"{node_name} manual index must contain only comma-separated integers.") from error
