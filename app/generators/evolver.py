import random
from typing import Dict, Any, List


class EvolInstructMutator:
    """Prompt mutator implementing the Evol-Instruct methodology (In-Depth & In-Breadth)."""

    EVOLUTION_TACTICS = [
        ("ADD_CONSTRAINTS", "Add the following strict constraint: the response must include numbered steps and a valid JSON format."),
        ("DEEPENING", "Deepen the question by requiring an explanation of underlying causes, associated risks, and mitigations."),
        ("CONCRETIFYING", "Apply the question to a concrete enterprise e-commerce scenario with real customers."),
        ("INCREASE_REASONING", "Require step-by-step reasoning (Chain-of-Thought) before providing the final conclusion."),
        ("IN_BREADTH", "Generate a new derived question on a related yet more complex topic within the same domain.")
    ]

    def mutate_prompt(self, base_instruction: str, tactic_name: str = None) -> Dict[str, Any]:
        """
        Applies a synthetic mutation to a base instruction.
        """
        if not tactic_name:
            tactic_name, prompt_modifier = random.choice(self.EVOLUTION_TACTICS)
        else:
            prompt_modifier = next((desc for name, desc in self.EVOLUTION_TACTICS if name == tactic_name), self.EVOLUTION_TACTICS[0][1])

        evolved_instruction = f"{base_instruction}\n\n[Synthetic Evolution Instruction: {prompt_modifier}]"

        return {
            "original_instruction": base_instruction,
            "tactic_used": tactic_name,
            "evolved_instruction": evolved_instruction,
            "complexity_level": "advanced"
        }
