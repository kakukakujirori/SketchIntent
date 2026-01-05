"""Pytest fixtures for img2cad_diffusion tests."""

import pytest


@pytest.fixture
def sequence_file() -> str:
    """Path to the SketchGraphs sequence data file."""
    return '/home/ryotaro/github/vitruvion/data_processed/sg_filtered_unique.npy'


@pytest.fixture
def fc_json_path() -> str:
    """Path to the fully constrained sketches JSON file."""
    return '/home/ryotaro/github/vitruvion/scripts/sketchgraphs_fc_search/sketchgraphs_fc_clustered.json'
