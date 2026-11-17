"""Selected explicitly on an exclusive TPU host; collection initializes no runtime."""
import pytest

from amora.backends.tpu.measurement import execute_recipe
from amora.backends.tpu.artifacts import validate_manifest


@pytest.mark.tpu
def test_tpu_compiled_vector_and_manifest(tmp_path):
    recipe = {"schema_version": 1, "recipe_id": "test-vector", "samples": 5,
              "repeats": 1, "max_relative_mad": 1,
              "cases": [{"probe_id": "vector.elementwise_throughput", "shape": [128,128]}]}
    root, manifest = execute_recipe(recipe, tmp_path)
    assert manifest["status"] == "passed", manifest
    assert validate_manifest(root/"manifest.json")["run_digest"] == manifest["run_digest"]


@pytest.mark.tpu_pallas
def test_tpu_pallas_execution(tmp_path):
    recipe = {"schema_version": 1, "recipe_id": "test-pallas", "samples": 5,
              "repeats": 1, "max_relative_mad": 1,
              "cases": [{"probe_id": "global_memory.streaming", "shape": [128,128], "tile": [128,128]}]}
    _, manifest = execute_recipe(recipe, tmp_path)
    assert manifest["status"] == "passed", manifest
