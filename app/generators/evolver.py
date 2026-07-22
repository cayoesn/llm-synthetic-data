import random
from typing import Dict, Any, List


class EvolInstructMutator:
    """Mutador de Prompts usando a Metodologia Evol-Instruct (In-Depth & In-Breadth)."""

    EVOLUTION_TACTICS = [
        ("ADD_CONSTRAINTS", "Adicione a seguinte restricao estrita: a resposta deve incluir etapas numeradas e formato JSON validavel."),
        ("DEEPENING", "Aprofunde a pergunta exigindo explicacao dos motivos subjacentes, riscos associados e mitigacoes."),
        ("CONCRETIFYING", "Aplica a pergunta a um cenario corporativo concreto de e-commerce com clientes reais."),
        ("INCREASE_REASONING", "Exija um raciocinio passo a passo (Chain-of-Thought) antes de fornecer a conclusao final."),
        ("IN_BREADTH", "Gere uma nova pergunta derivada sobre um tema correlato mas mais complexo no mesmo dominio.")
    ]

    def mutate_prompt(self, base_instruction: str, tactic_name: str = None) -> Dict[str, Any]:
        """
        Aplica uma mutação sintética a uma instrução base.
        """
        if not tactic_name:
            tactic_name, prompt_modifier = random.choice(self.EVOLUTION_TACTICS)
        else:
            prompt_modifier = next((desc for name, desc in self.EVOLUTION_TACTICS if name == tactic_name), self.EVOLUTION_TACTICS[0][1])

        evolved_instruction = f"{base_instruction}\n\n[Instrucao de Evolucao Sintetica: {prompt_modifier}]"

        return {
            "original_instruction": base_instruction,
            "tactic_used": tactic_name,
            "evolved_instruction": evolved_instruction,
            "complexity_level": "advanced"
        }
