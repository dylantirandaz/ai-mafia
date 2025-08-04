# prompt_engine.py

from typing import List, Dict
import json

class MemoryStore:
    """Naïve in‑memory event store + keyword retrieval."""
    def __init__(self):
        self.events: List[str] = []
    def add_event(self, description: str):
        self.events.append(description)
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        kws = query.lower().split()
        hits = [e for e in self.events if any(w in e.lower() for w in kws)]
        return hits[-k:]


class PromptEngine:
    """Assembles prompts with role, CoT template, and memory snippets."""
    COT_HEADER = "Think step-by-step before choosing your action:\n"
    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def build_action_prompt(
        self,
        player_id: str,
        role: str,
        phase: str,
        game_state: Dict,
        legal_actions: List[Dict]
    ) -> str:
        state = json.dumps({
            "phase": phase,
            "round": game_state["round_number"],
            "alive": game_state["get_alive_players"](),
            "last_events": game_state["last_events"][-3:]
        }, indent=2)
        mem_snippets = self.memory.retrieve(" ".join(game_state["last_events"]))
        mem_block = "\n".join(f"- {m}" for m in mem_snippets)
        cot = (
            f"{self.COT_HEADER}"
            f"As a {role.upper()}, you can {', '.join(a['action'] for a in legal_actions)}.\n"
            f"Game state:\n{state}\n"
        )
        if mem_block:
            cot += f"Relevant memory:\n{mem_block}\n"
        cot += "Now output a JSON object `{action:..., target:...}`.\n"
        return cot

    def build_chat_prompt(
        self,
        player_id: str,
        role: str,
        game_state: Dict,
        discussion_history: List[Dict]
    ) -> str:
        last_msgs = "\n".join(
            f"{m['player']}: {m['message']}" 
            for m in discussion_history[-5:]
        )
        mem = self.memory.retrieve(last_msgs)
        cot = f"{self.COT_HEADER}You are {player_id} ({role.upper()}).\n"
        cot += f"Recent chat:\n{last_msgs}\n"
        if mem:
            cot += "Memory:\n" + "\n".join(f"- {m}" for m in mem) + "\n"
        cot += "Now write a concise but strategic message."
        return cot
