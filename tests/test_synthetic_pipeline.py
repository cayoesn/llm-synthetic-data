import os
import pytest
from app.generators.evolver import EvolInstructMutator
from app.generators.synthetic_pipeline import SyntheticDataPipeline
from app.finetuning.qlora_trainer import QLoRATrainerPipeline
from app.observability import synth_obs


def test_evolver_mutations():
    mutator = EvolInstructMutator()
    res = mutator.mutate_prompt("Como responder a uma reclamacao?", tactic_name="ADD_CONSTRAINTS")
    assert res["tactic_used"] == "ADD_CONSTRAINTS"
    assert "JSON" in res["evolved_instruction"]


def test_synthetic_pipeline():
    pipeline = SyntheticDataPipeline()
    dataset = pipeline.generate_dataset(num_samples_per_seed=2)
    
    assert len(dataset) == 8  # 4 seed prompts x 2
    assert "instruction" in dataset[0]
    assert "response" in dataset[0]

    output_file = os.path.join(os.path.dirname(__file__), "..", "data", "test_dataset.json")
    saved_path = pipeline.export_dataset_json(dataset, output_file)
    assert os.path.exists(saved_path)

    # Observabilidade
    synth_obs.trace_dataset_generation(len(dataset), ["Atendimento", "DevOps"])


def test_qlora_finetuning_trainer():
    trainer = QLoRATrainerPipeline(r=16, alpha=32)
    lora_conf = trainer.configure_lora_config()
    
    assert lora_conf["r"] == 16
    assert lora_conf["lora_alpha"] == 32
    assert "q_proj" in lora_conf["target_modules"]

    dummy_dataset = [{"instruction": "test", "response": "test"}] * 10
    train_res = trainer.train(dummy_dataset)

    assert train_res["status"] == "COMPLETED"
    assert train_res["final_loss"] < 2.0
    assert train_res["samples_trained"] == 10

    # Observabilidade
    synth_obs.trace_finetuning_run(train_res)
