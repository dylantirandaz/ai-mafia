import asyncio
import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Role(Enum):
    INNOCENT = "innocent"
    KILLER = "killer"
    DETECTIVE = "detective"  # Optional special role
    DOCTOR = "doctor"  # Optional special role

class Phase(Enum):
    NIGHT = "night"
    DISCUSSION = "discussion"
    VOTING = "voting"
    GAME_OVER = "game_over"

class ActionType(Enum):
    KILL = "kill"
    INVESTIGATE = "investigate"
    PROTECT = "protect"
    VOTE = "vote"
    DISCUSS = "discuss"

@dataclass
class PlayerState:
    """Represents the state of a player in the game"""
    player_id: str
    role: Role
    alive: bool = True
    protected: bool = False
    votes_received: int = 0
    voting_for: Optional[str] = None
    
@dataclass
class GameState:
    """Complete game state"""
    players: Dict[str, PlayerState]
    phase: Phase
    round_number: int
    discussion_history: List[Dict[str, Any]] = field(default_factory=list)
    night_actions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_eliminated: List[str] = field(default_factory=list)
    last_killed: List[str] = field(default_factory=list)
    investigation_results: Dict[str, List[Tuple[str, Role]]] = field(default_factory=dict)
    
    def get_alive_players(self) -> List[str]:
        return [pid for pid, p in self.players.items() if p.alive]
    
    def get_players_by_role(self, role: Role) -> List[str]:
        return [pid for pid, p in self.players.items() if p.role == role and p.alive]

class ModelAdapter(ABC):
    """Abstract base class for different AI model adapters"""
    
    @abstractmethod
    async def get_action(self, 
                        game_state: GameState, 
                        player_id: str, 
                        phase: Phase,
                        legal_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get an action from the AI model"""
        pass
    
    @abstractmethod
    async def generate_discussion(self,
                                 game_state: GameState,
                                 player_id: str,
                                 discussion_context: List[Dict[str, Any]]) -> str:
        """Generate a discussion message"""
        pass

class PromptBuilder:
    """Builds prompts for different AI models and game situations"""
    
    @staticmethod
    def build_role_prompt(player_id: str, role: Role, game_state: GameState) -> str:
        """Build initial role assignment prompt"""
        alive_players = game_state.get_alive_players()
        
        base_prompt = f"""You are playing a game of Mafia/Among Us as Player {player_id}.

GAME RULES:
- There are {len(alive_players)} players alive: {', '.join(alive_players)}
- Killers win if they equal or outnumber innocents
- Innocents win by eliminating all killers
- Each round has: Night phase (killers act) → Discussion phase → Voting phase
- Players vote to eliminate someone each round
- Eliminated players cannot participate

YOUR SECRET ROLE: {role.value.upper()}"""

        role_specific = {
            Role.KILLER: """
As a KILLER:
- You must secretly eliminate innocents
- Coordinate with other killers (if any) without revealing yourself
- Blend in during discussions to avoid suspicion
- Your goal is to survive until killers equal/outnumber innocents""",
            
            Role.INNOCENT: """
As an INNOCENT:
- You must identify and vote out the killers
- Pay attention to suspicious behavior and voting patterns
- Share information to help other innocents
- Your goal is to eliminate all killers""",
            
            Role.DETECTIVE: """
As a DETECTIVE (special innocent):
- You can investigate one player each night to learn their role
- Use your investigations wisely to find killers
- Share information strategically without revealing your special role
- Help innocents identify killers""",
            
            Role.DOCTOR: """
As a DOCTOR (special innocent):
- You can protect one player each night from being killed
- You cannot protect yourself
- Choose protection targets strategically
- Help innocents while keeping your role hidden"""
        }
        
        return base_prompt + role_specific.get(role, "")
    
    @staticmethod
    def build_night_action_prompt(player_id: str, role: Role, game_state: GameState) -> str:
        """Build prompt for night phase actions"""
        alive_players = [p for p in game_state.get_alive_players() if p != player_id]
        
        prompts = {
            Role.KILLER: f"""NIGHT PHASE - KILLER ACTION
Choose a player to eliminate. Living players: {', '.join(alive_players)}
Respond with JSON: {{"action": "kill", "target": "PlayerX"}}""",
            
            Role.DETECTIVE: f"""NIGHT PHASE - DETECTIVE ACTION
Choose a player to investigate. Living players: {', '.join(alive_players)}
Previous investigations: {game_state.investigation_results.get(player_id, [])}
Respond with JSON: {{"action": "investigate", "target": "PlayerX"}}""",
            
            Role.DOCTOR: f"""NIGHT PHASE - DOCTOR ACTION
Choose a player to protect (not yourself). Living players: {', '.join(alive_players)}
Respond with JSON: {{"action": "protect", "target": "PlayerX"}}""",
            
            Role.INNOCENT: """NIGHT PHASE - INNOCENT
You have no special night action. Waiting for day phase.
Respond with JSON: {"action": "wait"}"""
        }
        
        return prompts.get(role, prompts[Role.INNOCENT])
    
    @staticmethod
    def build_discussion_prompt(player_id: str, role: Role, game_state: GameState, 
                               model_type: str = "default") -> str:
        """Build prompt for discussion phase"""
        
        # Get recent events
        events = []
        if game_state.last_killed:
            events.append(f"Killed last night: {', '.join(game_state.last_killed)}")
        if game_state.last_eliminated:
            events.append(f"Voted out last round: {', '.join(game_state.last_eliminated)}")
        
        # Get discussion history (last 10 messages)
        recent_discussion = game_state.discussion_history[-10:] if game_state.discussion_history else []
        
        # Model-specific formatting
        if model_type == "claude":
            prompt = f"""DISCUSSION PHASE - Round {game_state.round_number}

You are Player {player_id} (secretly {role.value}).
Alive players: {', '.join(game_state.get_alive_players())}

Recent events:
{chr(10).join(events) if events else "No recent deaths"}

Recent discussion:
{chr(10).join([f"{msg['player']}: {msg['message']}" for msg in recent_discussion])}

Generate a strategic message (1-3 sentences) that:
- Advances your role's objectives
- Responds to the current situation
- Maintains your cover if you're a killer

Remember: Be concise and strategic. Don't reveal your role directly."""
        
        elif model_type == "gpt4":
            prompt = f"""[MAFIA GAME - DISCUSSION PHASE]
Player: {player_id} | Secret Role: {role.value} | Round: {game_state.round_number}

CURRENT SITUATION:
• Living players: {', '.join(game_state.get_alive_players())}
• {events[0] if events else "No deaths yet"}

RECENT DISCUSSION:
{chr(10).join([f"- {msg['player']}: {msg['message']}" for msg in recent_discussion[-5:]])}

YOUR TASK: Write a strategic message (max 2 sentences) that helps your team win.
If killer: Deflect suspicion. If innocent: Share suspicions or analysis."""
        
        else:  # Default format
            prompt = f"""You are Player {player_id} in a Mafia game. Your secret role: {role.value}

Game State:
- Round {game_state.round_number}
- Alive: {', '.join(game_state.get_alive_players())}
- Recent: {'; '.join(events) if events else 'Game just started'}

Discussion History:
{chr(10).join([f"{msg['player']}: {msg['message']}" for msg in recent_discussion[-7:]])}

Write a strategic message to influence the game in your favor:"""
        
        return prompt
    
    @staticmethod
    def build_voting_prompt(player_id: str, role: Role, game_state: GameState,
                           voting_history: List[Dict[str, str]] = None) -> str:
        """Build prompt for voting phase"""
        alive_players = [p for p in game_state.get_alive_players() if p != player_id]
        
        # Summarize discussion
        discussion_summary = []
        for msg in game_state.discussion_history[-15:]:
            if any(word in msg['message'].lower() for word in ['suspect', 'think', 'vote', 'killer', 'suspicious']):
                discussion_summary.append(f"{msg['player']}: {msg['message']}")
        
        prompt = f"""VOTING PHASE - Round {game_state.round_number}

You are Player {player_id} (secretly {role.value}).
You must vote to eliminate someone.

Living players you can vote for: {', '.join(alive_players)}

Key discussion points:
{chr(10).join(discussion_summary[-5:]) if discussion_summary else "No strong accusations made"}

Previous voting patterns:
{json.dumps(voting_history[-3:], indent=2) if voting_history else "No previous votes"}

STRATEGY REMINDER:
- Killers: Vote to eliminate innocents or create confusion
- Innocents: Vote to eliminate suspected killers
- Consider: Who's been suspicious? Who's deflecting? Who's too quiet?

Respond with JSON: {{"action": "vote", "target": "PlayerX", "reasoning": "Brief explanation"}}"""
        
        return prompt

class MafiaGameEngine:
    """Main game engine that manages game flow and rules"""
    
    def __init__(self, num_players: int = 6, num_killers: int = 2, 
                 enable_special_roles: bool = False):
        self.num_players = num_players
        self.num_killers = num_killers
        self.enable_special_roles = enable_special_roles
        self.game_state: Optional[GameState] = None
        self.model_adapters: Dict[str, ModelAdapter] = {}
        self.game_log: List[Dict[str, Any]] = []
        self.voting_history: List[Dict[str, str]] = []
        
    def initialize_game(self) -> GameState:
        """Initialize a new game with random role assignments"""
        players = {}
        player_ids = [f"Player{i}" for i in range(1, self.num_players + 1)]
        
        # Shuffle for random assignment
        random.shuffle(player_ids)
        
        # Assign roles
        roles_assigned = 0
        
        # Assign killers
        for i in range(self.num_killers):
            players[player_ids[roles_assigned]] = PlayerState(
                player_id=player_ids[roles_assigned],
                role=Role.KILLER
            )
            roles_assigned += 1
        
        # Assign special roles if enabled
        if self.enable_special_roles and roles_assigned < len(player_ids) - 1:
            # Detective
            players[player_ids[roles_assigned]] = PlayerState(
                player_id=player_ids[roles_assigned],
                role=Role.DETECTIVE
            )
            roles_assigned += 1
            
            # Doctor (if enough players)
            if roles_assigned < len(player_ids) - 1:
                players[player_ids[roles_assigned]] = PlayerState(
                    player_id=player_ids[roles_assigned],
                    role=Role.DOCTOR
                )
                roles_assigned += 1
        
        # Rest are innocents
        for i in range(roles_assigned, len(player_ids)):
            players[player_ids[i]] = PlayerState(
                player_id=player_ids[i],
                role=Role.INNOCENT
            )
        
        self.game_state = GameState(
            players=players,
            phase=Phase.NIGHT,
            round_number=1
        )
        
        self._log_event("game_initialized", {
            "players": {pid: p.role.value for pid, p in players.items()},
            "num_killers": self.num_killers
        })
        
        return self.game_state
    
    def register_model_adapter(self, player_id: str, adapter: ModelAdapter):
        """Register a model adapter for a player"""
        self.model_adapters[player_id] = adapter
    
    async def run_night_phase(self) -> Dict[str, Any]:
        """Execute night phase actions"""
        logger.info(f"Starting night phase - Round {self.game_state.round_number}")
        
        night_actions = {}
        
        # Collect actions from all players with night abilities
        for player_id, player in self.game_state.players.items():
            if not player.alive:
                continue
                
            if player.role in [Role.KILLER, Role.DETECTIVE, Role.DOCTOR]:
                adapter = self.model_adapters.get(player_id)
                if adapter:
                    try:
                        action = await adapter.get_action(
                            self.game_state, 
                            player_id, 
                            Phase.NIGHT,
                            self._get_legal_actions(player_id, Phase.NIGHT)
                        )
                        night_actions[player_id] = action
                        logger.info(f"{player_id} ({player.role.value}) performed action: {action}")
                    except Exception as e:
                        logger.error(f"Error getting action from {player_id}: {e}")
        
        # Process actions
        results = self._process_night_actions(night_actions)
        
        self._log_event("night_phase_complete", {
            "round": self.game_state.round_number,
            "actions": night_actions,
            "results": results
        })
        
        return results
    
    async def run_discussion_phase(self, duration_seconds: int = 60) -> List[Dict[str, Any]]:
        """Run discussion phase where players communicate"""
        logger.info("Starting discussion phase")
        self.game_state.phase = Phase.DISCUSSION
        
        discussion_messages = []
        start_time = time.time()
        
        # Multiple rounds of discussion within time limit
        while time.time() - start_time < duration_seconds:
            for player_id, player in self.game_state.players.items():
                if not player.alive:
                    continue
                
                adapter = self.model_adapters.get(player_id)
                if adapter:
                    try:
                        message = await adapter.generate_discussion(
                            self.game_state,
                            player_id,
                            discussion_messages
                        )
                        
                        discussion_entry = {
                            "player": player_id,
                            "message": message,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        discussion_messages.append(discussion_entry)
                        self.game_state.discussion_history.append(discussion_entry)
                        
                        logger.info(f"{player_id}: {message}")
                        
                    except Exception as e:
                        logger.error(f"Error getting discussion from {player_id}: {e}")
            
            # Small delay between discussion rounds
            await asyncio.sleep(2)
        
        self._log_event("discussion_phase_complete", {
            "round": self.game_state.round_number,
            "messages": discussion_messages
        })
        
        return discussion_messages
    
    async def run_voting_phase(self) -> Dict[str, Any]:
        """Run voting phase where players vote to eliminate someone"""
        logger.info("Starting voting phase")
        self.game_state.phase = Phase.VOTING
        
        votes = {}
        
        # Reset previous votes
        for player in self.game_state.players.values():
            player.votes_received = 0
            player.voting_for = None
        
        # Collect votes
        for player_id, player in self.game_state.players.items():
            if not player.alive:
                continue
            
            adapter = self.model_adapters.get(player_id)
            if adapter:
                try:
                    vote_action = await adapter.get_action(
                        self.game_state,
                        player_id,
                        Phase.VOTING,
                        self._get_legal_actions(player_id, Phase.VOTING)
                    )
                    
                    target = vote_action.get("target")
                    if target and target in self.game_state.players and self.game_state.players[target].alive:
                        votes[player_id] = target
                        player.voting_for = target
                        self.game_state.players[target].votes_received += 1
                        
                        logger.info(f"{player_id} votes for {target}: {vote_action.get('reasoning', 'No reason given')}")
                        
                except Exception as e:
                    logger.error(f"Error getting vote from {player_id}: {e}")
        
        # Tally votes and eliminate player with most votes
        elimination_result = self._process_votes(votes)
        
        self.voting_history.append({
            "round": self.game_state.round_number,
            "votes": votes,
            "eliminated": elimination_result["eliminated"]
        })
        
        self._log_event("voting_phase_complete", {
            "round": self.game_state.round_number,
            "votes": votes,
            "result": elimination_result
        })
        
        return elimination_result
    
    def check_win_condition(self) -> Optional[str]:
        """Check if game has ended"""
        alive_killers = len(self.game_state.get_players_by_role(Role.KILLER))
        alive_innocents = len([p for p in self.game_state.get_alive_players() 
                              if self.game_state.players[p].role != Role.KILLER])
        
        if alive_killers == 0:
            return "innocents"
        elif alive_killers >= alive_innocents:
            return "killers"
        
        return None
    
    def _get_legal_actions(self, player_id: str, phase: Phase) -> List[Dict[str, Any]]:
        """Get legal actions for a player in current phase"""
        player = self.game_state.players[player_id]
        alive_others = [p for p in self.game_state.get_alive_players() if p != player_id]
        
        if phase == Phase.NIGHT:
            if player.role == Role.KILLER:
                return [{"action": "kill", "targets": alive_others}]
            elif player.role == Role.DETECTIVE:
                investigated = [target for target, _ in self.game_state.investigation_results.get(player_id, [])]
                uninvestigated = [p for p in alive_others if p not in investigated]
                return [{"action": "investigate", "targets": uninvestigated}]
            elif player.role == Role.DOCTOR:
                return [{"action": "protect", "targets": alive_others}]
        
        elif phase == Phase.VOTING:
            return [{"action": "vote", "targets": alive_others}]
        
        return []
    
    def _process_night_actions(self, actions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Process all night actions and return results"""
        results = {
            "killed": [],
            "protected": [],
            "investigations": {}
        }
        
        # Process protections first
        for player_id, action in actions.items():
            if action.get("action") == "protect":
                target = action.get("target")
                if target:
                    self.game_state.players[target].protected = True
                    results["protected"].append(target)
        
        # Process kills
        kill_targets = []
        for player_id, action in actions.items():
            if action.get("action") == "kill":
                target = action.get("target")
                if target:
                    kill_targets.append(target)
        
        # Apply kills (majority or random if tie)
        if kill_targets:
            from collections import Counter
            kill_counts = Counter(kill_targets)
            max_votes = max(kill_counts.values())
            targets_with_max = [t for t, c in kill_counts.items() if c == max_votes]
            
            final_target = random.choice(targets_with_max)
            
            if not self.game_state.players[final_target].protected:
                self.game_state.players[final_target].alive = False
                results["killed"].append(final_target)
                self.game_state.last_killed = [final_target]
        
        # Process investigations
        for player_id, action in actions.items():
            if action.get("action") == "investigate":
                target = action.get("target")
                if target:
                    target_role = self.game_state.players[target].role
                    if player_id not in self.game_state.investigation_results:
                        self.game_state.investigation_results[player_id] = []
                    self.game_state.investigation_results[player_id].append((target, target_role))
                    results["investigations"][player_id] = (target, target_role.value)
        
        # Reset protections
        for player in self.game_state.players.values():
            player.protected = False
        
        return results
    
    def _process_votes(self, votes: Dict[str, str]) -> Dict[str, Any]:
        """Process votes and eliminate player"""
        if not votes:
            return {"eliminated": None, "tie": True}
        
        # Count votes
        from collections import Counter
        vote_counts = Counter(votes.values())
        
        if not vote_counts:
            return {"eliminated": None, "tie": True}
        
        max_votes = max(vote_counts.values())
        candidates = [player for player, count in vote_counts.items() if count == max_votes]
        
        # Handle ties
        if len(candidates) > 1:
            # Could implement tie-breaking rules here
            eliminated = random.choice(candidates)
            tie = True
        else:
            eliminated = candidates[0]
            tie = False
        
        # Eliminate player
        self.game_state.players[eliminated].alive = False
        self.game_state.last_eliminated = [eliminated]
        
        return {
            "eliminated": eliminated,
            "votes_received": max_votes,
            "tie": tie,
            "vote_details": dict(vote_counts)
        }
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log game events for analysis"""
        self.game_log.append({
            "timestamp": datetime.now().isoformat(),
            "round": self.game_state.round_number if self.game_state else 0,
            "event_type": event_type,
            "data": data
        })
    
    async def run_game(self) -> Dict[str, Any]:
        """Run a complete game"""
        self.initialize_game()
        
        # Send initial role prompts to all players
        for player_id, player in self.game_state.players.items():
            logger.info(f"{player_id} assigned role: {player.role.value}")
        
        # Game loop
        while True:
            # Night phase
            night_results = await self.run_night_phase()
            
            # Check if game ended
            if winner := self.check_win_condition():
                self.game_state.phase = Phase.GAME_OVER
                break
            
            # Discussion phase
            await self.run_discussion_phase()
            
            # Voting phase
            voting_results = await self.run_voting_phase()
            
            # Check if game ended
            if winner := self.check_win_condition():
                self.game_state.phase = Phase.GAME_OVER
                break
            
            # Advance round
            self.game_state.round_number += 1
            self.game_state.phase = Phase.NIGHT
        
        # Game over
        game_result = {
            "winner": winner,
            "rounds_played": self.game_state.round_number,
            "final_survivors": self.game_state.get_alive_players(),
            "role_distribution": {pid: p.role.value for pid, p in self.game_state.players.items()},
            "game_log": self.game_log
        }
        
        self._log_event("game_over", game_result)
        
        return game_result