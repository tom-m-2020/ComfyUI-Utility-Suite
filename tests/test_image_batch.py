import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "utility_suite_test_package",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
IMAGE_BATCH = sys.modules[f"{SPEC.name}.image_batch"]


def test_batch_to_list_preserves_order_shape_dtype_and_storage_views():
    image = torch.arange(3 * 2 * 4 * 2, dtype=torch.float32).reshape(3, 2, 4, 2)

    images = IMAGE_BATCH.image_batch_to_list(image)

    assert len(images) == 3
    assert all(item.shape == (1, 2, 4, 2) for item in images)
    assert all(item.dtype == image.dtype and item.device == image.device for item in images)
    assert all(item.untyped_storage().data_ptr() == image.untyped_storage().data_ptr() for item in images)
    assert torch.equal(torch.cat(images), image)


def test_list_to_batch_flattens_element_batches_in_list_order():
    first = torch.full((2, 3, 4, 1), 1.0)
    second = torch.full((1, 3, 4, 1), 2.0)

    result = IMAGE_BATCH.image_list_to_batch([first, second])

    assert result.shape == (3, 3, 4, 1)
    assert torch.equal(result[:2], first)
    assert torch.equal(result[2:], second)


def test_single_item_list_returns_original_tensor():
    image = torch.rand((2, 3, 4, 3))
    assert IMAGE_BATCH.image_list_to_batch([image]) is image


def test_batch_list_round_trip_is_exact():
    image = torch.rand((4, 5, 7, 3), dtype=torch.float32)
    result = IMAGE_BATCH.image_list_to_batch(IMAGE_BATCH.image_batch_to_list(image))
    assert torch.equal(result, image)


@pytest.mark.parametrize(
    ("images", "message"),
    [
        ([], "non-empty"),
        ([torch.zeros(2, 3, 4)], "rank-4"),
        ([torch.zeros(0, 3, 4, 1)], "at least one"),
        ([torch.zeros(1, 3, 4, 1), torch.zeros(1, 3, 5, 1)], "common spatial"),
        ([torch.zeros(1, 3, 4, 1), torch.zeros(1, 3, 4, 2)], "common spatial"),
        ([torch.zeros(1, 3, 4, 1), torch.zeros(1, 3, 4, 1, dtype=torch.float16)], "dtype"),
    ],
)
def test_list_to_batch_rejects_incompatible_inputs(images, message):
    with pytest.raises(ValueError, match=message):
        IMAGE_BATCH.image_list_to_batch(images)


@pytest.mark.parametrize("image", [torch.zeros(3, 4, 1), torch.zeros(0, 3, 4, 1)])
def test_batch_to_list_rejects_invalid_inputs(image):
    with pytest.raises(ValueError):
        IMAGE_BATCH.image_batch_to_list(image)


def test_nodes_2_schemas_expose_list_transport():
    to_batch = IMAGE_BATCH.ImageListToImageBatch.define_schema()
    to_list = IMAGE_BATCH.ImageBatchToImageList.define_schema()
    assert to_batch.is_input_list is True
    assert to_list.outputs[0].is_output_list is True


def test_package_entrypoint_registers_all_nodes():
    extension = asyncio.run(PACKAGE.comfy_entrypoint())
    node_ids = {node.define_schema().node_id for node in asyncio.run(extension.get_node_list())}
    assert node_ids == {
        "UtilitySuiteImageTileBatch",
        "UtilitySuiteImageUntileBatch",
        "UtilitySuiteImageListToImageBatch",
        "UtilitySuiteImageBatchToImageList",
    }
