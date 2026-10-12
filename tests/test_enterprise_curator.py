import pytest
from app.generators.curator import DatasetCurator, DatasetSample, MinHashLSH, dataset_curator


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


def test_minhash_lsh_subquadratic_deduplication():
    """Valida a precisão da detecção de duplicatas do MinHash LSH."""
    lsh = MinHashLSH(num_perm=64, num_bands=16)

    doc1 = "Implementacao de algoritmo de ordenacao rapida em Python com complexidade n log n"
    doc2 = "Implementacao de algoritmo de ordenacao rapida em Python com complexidade n log n para arrays"
    doc3 = "Arquitetura distribuida de microsservicos com mensageria Kafka e gRPC em Go"

    sig1 = lsh.compute_signature(doc1)
    sig2 = lsh.compute_signature(doc2)
    sig3 = lsh.compute_signature(doc3)

    lsh.insert(sig1, 0)
    # doc2 deve colidir com doc1 com alta similaridade estimada
    assert lsh.is_duplicate(sig2, threshold=0.7) is True
    # doc3 é totalmente diferente e não deve ser duplicata
    assert lsh.is_duplicate(sig3, threshold=0.7) is False


def test_anti_hallucination_loop_filter():
    """Valida que textos com repetição degenerada são filtrados."""
    curator = DatasetCurator()
    repetitive_output = "Isto eh excelente. Isto eh excelente. Isto eh excelente. Isto eh excelente."
    assert curator.detect_repetitive_loop(repetitive_output) is True

    normal_output = "Isto eh excelente para producao porque reduz o custo computacional e melhora a precisao."
    assert curator.detect_repetitive_loop(normal_output) is False

    samples = [
        DatasetSample(id="1", instruction="Instrucao normal", output=repetitive_output),
        DatasetSample(id="2", instruction="Instrucao normal 2", output=normal_output),
    ]
    clean = curator.deduplicate_and_filter(samples, similarity_threshold=0.8, min_output_length=10)
    assert len(clean) == 1
    assert clean[0].id == "2"
