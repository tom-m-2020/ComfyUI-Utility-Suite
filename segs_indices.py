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


def validate_segs_indices(index: object, count: int, node_name: str) -> list[int]:
    if not isinstance(index, list):
        raise TypeError(f"{node_name} index must be a list of integers.")

    seen: set[int] = set()
    for position, seg_index in enumerate(index):
        if isinstance(seg_index, bool) or not isinstance(seg_index, int):
            raise TypeError(
                f"{node_name} index[{position}] must be an integer, got {type(seg_index).__name__}."
            )
        if seg_index < 0 or seg_index >= count:
            raise ValueError(
                f"{node_name} index[{position}]={seg_index} is outside the valid SEG range 0..{count - 1}."
            )
        if seg_index in seen:
            raise ValueError(f"{node_name} index contains duplicate SEG position {seg_index}.")
        seen.add(seg_index)
    return index
