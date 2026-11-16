from __future__ import annotations

import json
import secrets

from comfy_api.latest import io

MAX_SEED = 1125899906842624


def parse_seed_state(state: str) -> tuple[int, bool]:
    try:
        value = json.loads(state)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Seed Generator state must be a JSON object.") from error
    if not isinstance(value, dict):
        raise TypeError("Seed Generator state must be a JSON object.")

    seed = value.get("seed", 0)
    randomize = value.get("randomize", True)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("Seed Generator seed must be an integer.")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"Seed Generator seed must be within 0..{MAX_SEED}.")
    if not isinstance(randomize, bool):
        raise TypeError("Seed Generator randomize must be boolean.")
    return seed, randomize


def generate_seed(state: str) -> int:
    seed, randomize = parse_seed_state(state)
    return secrets.randbelow(MAX_SEED + 1) if randomize else seed


class SeedGenerator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UtilitySuiteSeedGenerator",
            display_name="Seed Generator",
            category="Utility Suite/Utilities",
            description="Produces fixed or server-randomized seeds without mutating queued prompt data.",
            inputs=[
                io.String.Input(
                    "state",
                    default='{"seed":0,"randomize":true}',
                    extra_dict={"seed_generator": True},
                ),
            ],
            outputs=[io.Int.Output("seed")],
        )

    @classmethod
    def fingerprint_inputs(cls, state: str):
        seed, randomize = parse_seed_state(state)
        return secrets.token_hex(16) if randomize else seed

    @classmethod
    def execute(cls, state: str) -> io.NodeOutput:
        seed = generate_seed(state)
        return io.NodeOutput(seed, ui={"seed": [seed]})


__all__ = ["MAX_SEED", "SeedGenerator", "generate_seed", "parse_seed_state"]
