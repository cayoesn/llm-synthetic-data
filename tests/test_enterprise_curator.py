import pytest
from app.generators.curator import DatasetCurator, DatasetSample, MinHashLSH, dataset_curator


def test_jaccard_similarity():
    sim = dataset_curator.jaccard_similarity(
        "How to build an AI model?",
        "How to quickly build an AI model?"
    )
    assert sim > 0.6


def test_deduplicate_and_filter():
    curator = DatasetCurator()
    samples = [
        DatasetSample(id="1", instruction="What is Python?", output="A programming language."),
        DatasetSample(id="2", instruction="What is Python?", output="A very popular programming language."),
        DatasetSample(id="3", instruction="What is Rust?", output="A fast systems programming language."),
        DatasetSample(id="4", instruction="Invalid", output="short"),
    ]

    clean = curator.deduplicate_and_filter(samples, similarity_threshold=0.8, min_output_length=10)
    assert len(clean) == 2
    assert clean[0].id == "1"
    assert clean[1].id == "3"


def test_minhash_lsh_subquadratic_deduplication():
    """Validates duplicate detection accuracy of MinHash LSH."""
    lsh = MinHashLSH(num_perm=64, num_bands=16)

    doc1 = "Implementation of the quicksort algorithm in Python with n log n time complexity"
    doc2 = "Implementation of the quicksort algorithm in Python with n log n time complexity for arrays"
    doc3 = "Distributed microservices architecture using Kafka messaging and gRPC in Go"

    sig1 = lsh.compute_signature(doc1)
    sig2 = lsh.compute_signature(doc2)
    sig3 = lsh.compute_signature(doc3)

    lsh.insert(sig1, 0)
    # doc2 should collide with doc1 with high estimated similarity
    assert lsh.is_duplicate(sig2, threshold=0.7) is True
    # doc3 is completely different and must not be flagged as a duplicate
    assert lsh.is_duplicate(sig3, threshold=0.7) is False


def test_anti_hallucination_loop_filter():
    """Validates that texts with degenerate repetition are filtered out."""
    curator = DatasetCurator()
    repetitive_output = "This is great. This is great. This is great. This is great."
    assert curator.detect_repetitive_loop(repetitive_output) is True

    normal_output = "This is great for production because it reduces compute cost and improves accuracy."
    assert curator.detect_repetitive_loop(normal_output) is False

    samples = [
        DatasetSample(id="1", instruction="Normal instruction", output=repetitive_output),
        DatasetSample(id="2", instruction="Normal instruction 2", output=normal_output),
    ]
    clean = curator.deduplicate_and_filter(samples, similarity_threshold=0.8, min_output_length=10)
    assert len(clean) == 1
    assert clean[0].id == "2"
