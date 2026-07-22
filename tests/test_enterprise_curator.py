import pytest
from app.generators.curator import DatasetCurator, DatasetSample, dataset_curator


def test_jaccard_similarity():
    sim = dataset_curator.jaccard_similarity(
        "Como criar um modelo de IA?",
        "Como podemos criar um modelo de IA?"
    )
    assert sim > 0.6


def test_deduplicate_and_filter():
    curator = DatasetCurator()
    samples = [
        DatasetSample(id="1", instruction="O que eh Python?", output="Linguagem de programação."),
        DatasetSample(id="2", instruction="O que eh Python?", output="Linguagem de programação muito popular."),
        DatasetSample(id="3", instruction="O que eh Rust?", output="Linguagem de sistemas rápida."),
        DatasetSample(id="4", instruction="Invalido", output="curto"),
    ]

    clean = curator.deduplicate_and_filter(samples, similarity_threshold=0.8, min_output_length=10)
    assert len(clean) == 2
    assert clean[0].id == "1"
    assert clean[1].id == "3"
