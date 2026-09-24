from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

_MISSING = object()


class RerouteOptional(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        value_type = io.MatchType.Template("value")
        return io.Schema(
            node_id="UtilitySuiteRerouteOptional",
            display_name="Reroute (Optional)",
            category="Utility Suite/Routing",
            inputs=[
                io.MatchType.Input(
                    "value",
                    template=value_type,
                    display_name="*",
                    optional=True,
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=value_type,
                    id="value",
                    display_name="*",
                ),
                io.Boolean.Output("present", display_name="present"),
            ],
        )

    @classmethod
    def execute(cls, value=_MISSING) -> io.NodeOutput:
        if value is _MISSING:
            return io.NodeOutput(None, False)
        return io.NodeOutput(value, True)


class RerouteOptionalExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [RerouteOptional]


async def comfy_entrypoint() -> RerouteOptionalExtension:
    return RerouteOptionalExtension()


__all__ = ["RerouteOptional", "RerouteOptionalExtension", "comfy_entrypoint"]
