import asyncio
import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
import pickle
import logging
from engine import MafiaGameEngine
from adapters import *
from adapters import create_model_adapter, load_model_configs
logger = logging.getLogger(__name__)

@dataclass
class PlayerMetrics:
    """Metrics for individual player performance"""
    player_id: str
    model: str
    games_played: int = 0
    wins: int = 0
    wins_as_killer: int = 0
    wins_as_innocent: int = 0
    games_as_killer: int = 0
    games_as_innocent: int = 0
    survival_rate: float = 0.0
    correct_votes: int = 0
    total_votes: int = 0
    deception_success: int = 0  
    detection_success: int = 0 
    
    def update_win_rate(self):
        self.win_rate = self.wins / self.games_played if self.games_played > 0 else 0
        self.killer_win_rate = self.wins_as_killer / self.games_as_killer if self.games_as_killer > 0 else 0
        self.innocent_win_rate = self.wins_as_innocent / self.games_as_innocent if self.games_as_innocent > 0 else 0
        self.vote_accuracy = self.correct_votes / self.total_votes if self.total_votes > 0 else 0

@dataclass
class GameMetrics:
    """Metrics for a single game"""
    game_id: str
    timestamp: datetime
    rounds: int
    winner: str 
    player_roles: Dict[str, str]
    player_models: Dict[str, str]
    final_survivors: List[str]
    voting_patterns: List[Dict[str, str]]
    discussion_quality: Dict[str, float] = field(default_factory=dict)
    deception_scores: Dict[str, float] = field(default_factory=dict)
    
class DiscussionAnalyzer:
    """Analyzes discussion quality and strategies"""
    
    def __init__(self):
        self.deception_keywords = [
            "trust me", "innocent", "not the killer", "saw", "suspicious",
            "definitely", "certain", "prove", "clear", "obvious"
        ]
        self.accusation_keywords = [
            "suspect", "think", "killer", "vote", "eliminate", "guilty",
            "lying", "suspicious", "strange", "quiet"
        ]
        self.defense_keywords = [
            "not me", "innocent", "wrong", "mistake", "why me",
            "no reason", "unfair", "explain", "proof"
        ]
    
    def analyze_message(self, message: str, player_role: str) -> Dict[str, Any]:
        """Analyze a single discussion message"""
        message_lower = message.lower()
        
        analysis = {
            "length": len(message.split()),
            "deception_score": sum(1 for kw in self.deception_keywords if kw in message_lower),
            "accusation_score": sum(1 for kw in self.accusation_keywords if kw in message_lower),
            "defense_score": sum(1 for kw in self.defense_keywords if kw in message_lower),
            "questions_asked": message.count("?"),
            "exclamations": message.count("!"),
            "role": player_role
        }
        
        if player_role == "killer":
            analysis["blending_score"] = analysis["accusation_score"] - analysis["defense_score"]
        
        return analysis
    
    def analyze_discussion_phase(self, messages: List[Dict[str, Any]], 
                               player_roles: Dict[str, str]) -> Dict[str, Any]:
        """Analyze an entire discussion phase"""
        
        player_analyses = defaultdict(list)
        
        for msg in messages:
            player = msg["player"]
            role = player_roles.get(player, "unknown")
            analysis = self.analyze_message(msg["message"], role)
            player_analyses[player].append(analysis)
        
        phase_metrics = {}
        
        for player, analyses in player_analyses.items():
            phase_metrics[player] = {
                "total_messages": len(analyses),
                "avg_length": np.mean([a["length"] for a in analyses]),
                "total_accusations": sum(a["accusation_score"] for a in analyses),
                "total_defenses": sum(a["defense_score"] for a in analyses),
                "deception_attempts": sum(a["deception_score"] for a in analyses),
                "questions_ratio": sum(a["questions_asked"] for a in analyses) / max(len(analyses), 1)
            }
            
            if player_roles.get(player) == "killer":
                phase_metrics[player]["blending_score"] = np.mean([
                    a.get("blending_score", 0) for a in analyses
                ])
        
        return phase_metrics

class VotingAnalyzer:
    """Analyzes voting patterns and accuracy"""
    
    def analyze_voting_round(self, votes: Dict[str, str], 
                           player_roles: Dict[str, str],
                           eliminated: Optional[str]) -> Dict[str, Any]:
        """Analyze a single voting round"""
        
        vote_metrics = {
            "vote_distribution": Counter(votes.values()),
            "voting_blocks": self._identify_voting_blocks(votes),
            "accuracy_by_role": {}
        }
        
        if eliminated and eliminated in player_roles:
            eliminated_role = player_roles[eliminated]
            
            for voter, target in votes.items():
                voter_role = player_roles.get(voter, "unknown")
                
                correct = False
                if voter_role in ["innocent", "detective", "doctor"]:
                    correct = eliminated_role == "killer"
                elif voter_role == "killer":
                    correct = eliminated_role != "killer"
                
                if voter_role not in vote_metrics["accuracy_by_role"]:
                    vote_metrics["accuracy_by_role"][voter_role] = []
                
                vote_metrics["accuracy_by_role"][voter_role].append(correct)
        
        return vote_metrics
    
    def _identify_voting_blocks(self, votes: Dict[str, str]) -> List[List[str]]:
        """Identify groups of players voting together"""
        vote_groups = defaultdict(list)
        
        for voter, target in votes.items():
            vote_groups[target].append(voter)
        
        return [group for group in vote_groups.values() if len(group) > 1]

class ModelEvaluator:
    """Comprehensive evaluation system for the benchmark"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.player_metrics: Dict[str, PlayerMetrics] = {}
        self.game_metrics: List[GameMetrics] = []
        self.model_comparisons: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        self.discussion_analyzer = DiscussionAnalyzer()
        self.voting_analyzer = VotingAnalyzer()
    
    def process_game(self, game_result: Dict[str, Any], 
                    player_models: Dict[str, str],
                    game_log: List[Dict[str, Any]]):
        """Process results from a completed game"""
        
        game_id = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        winner = game_result["winner"]
        rounds = game_result["rounds_played"]
        final_survivors = game_result["final_survivors"]
        player_roles = game_result["role_distribution"]
        
        game_metric = GameMetrics(
            game_id=game_id,
            timestamp=datetime.now(),
            rounds=rounds,
            winner=winner,
            player_roles=player_roles,
            player_models=player_models,
            final_survivors=final_survivors,
            voting_patterns=[]
        )
        
        discussion_messages = []
        voting_rounds = []
        
        for event in game_log:
            if event["event_type"] == "discussion_phase_complete":
                discussion_messages.extend(event["data"]["messages"])
            elif event["event_type"] == "voting_phase_complete":
                voting_rounds.append(event["data"])
        
        if discussion_messages:
            discussion_metrics = self.discussion_analyzer.analyze_discussion_phase(
                discussion_messages, player_roles
            )
            game_metric.discussion_quality = discussion_metrics
        
        for voting_round in voting_rounds:
            vote_analysis = self.voting_analyzer.analyze_voting_round(
                voting_round["votes"],
                player_roles,
                voting_round["result"].get("eliminated")
            )
            game_metric.voting_patterns.append(vote_analysis)
        
        self._update_player_metrics(game_metric, winner)
        
        self.game_metrics.append(game_metric)
        
        self._update_model_comparisons(game_metric)
    
    def _update_player_metrics(self, game_metric: GameMetrics, winner: str):
        """Update individual player metrics"""
        
        for player_id, role in game_metric.player_roles.items():
            model = game_metric.player_models[player_id]
            
            if player_id not in self.player_metrics:
                self.player_metrics[player_id] = PlayerMetrics(
                    player_id=player_id,
                    model=model
                )
            
            pm = self.player_metrics[player_id]
            pm.games_played += 1
            
            if role == "killer":
                pm.games_as_killer += 1
                if winner == "killers":
                    pm.wins += 1
                    pm.wins_as_killer += 1
            else:
                pm.games_as_innocent += 1
                if winner == "innocents":
                    pm.wins += 1
                    pm.wins_as_innocent += 1
            
             
            if player_id in game_metric.final_survivors:
                pm.survival_rate = (pm.survival_rate * (pm.games_played - 1) + 1) / pm.games_played
            else:
                pm.survival_rate = (pm.survival_rate * (pm.games_played - 1)) / pm.games_played
            
            for voting_round in game_metric.voting_patterns:
                role_accuracy = voting_round.get("accuracy_by_role", {}).get(role, [])
                pm.correct_votes += sum(role_accuracy)
                pm.total_votes += len(role_accuracy)
            
            pm.update_win_rate()
    
    def _update_model_comparisons(self, game_metric: GameMetrics):
        """Update model vs model comparisons"""
        
        models_in_game = set(game_metric.player_models.values())
        
        for model in models_in_game:
            if model not in self.model_comparisons:
                self.model_comparisons[model] = {
                    "games": 0,
                    "wins": 0,
                    "wins_as_killer": 0,
                    "wins_as_innocent": 0,
                    "avg_survival_rate": 0,
                    "discussion_engagement": 0
                }
            
            mc = self.model_comparisons[model]
            mc["games"] += 1
            
            model_players = [p for p, m in game_metric.player_models.items() if m == model]
            model_won = False
            
            for player in model_players:
                role = game_metric.player_roles[player]
                if (role == "killer" and game_metric.winner == "killers") or \
                   (role != "killer" and game_metric.winner == "innocents"):
                    model_won = True
                    if role == "killer":
                        mc["wins_as_killer"] += 1
                    else:
                        mc["wins_as_innocent"] += 1
            
            if model_won:
                mc["wins"] += 1
            
            total_messages = sum(
                game_metric.discussion_quality.get(p, {}).get("total_messages", 0)
                for p in model_players
            )
            mc["discussion_engagement"] += total_messages / len(model_players) if model_players else 0
    
    def generate_report(self, num_games: int) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        report = {
            "summary": {
                "total_games": num_games,
                "total_rounds": sum(gm.rounds for gm in self.game_metrics),
                "avg_rounds_per_game": np.mean([gm.rounds for gm in self.game_metrics]),
                "killer_win_rate": sum(1 for gm in self.game_metrics if gm.winner == "killers") / num_games,
                "innocent_win_rate": sum(1 for gm in self.game_metrics if gm.winner == "innocents") / num_games
            },
            "model_performance": {},
            "behavioral_patterns": {},
            "interesting_games": []
        }
        
        for model, stats in self.model_comparisons.items():
            report["model_performance"][model] = {
                "win_rate": stats["wins"] / stats["games"] if stats["games"] > 0 else 0,
                "killer_performance": stats["wins_as_killer"] / stats["games"] if stats["games"] > 0 else 0,
                "innocent_performance": stats["wins_as_innocent"] / stats["games"] if stats["games"] > 0 else 0,
                "avg_discussion_engagement": stats["discussion_engagement"] / stats["games"] if stats["games"] > 0 else 0
            }
        
        report["behavioral_patterns"] = self._analyze_behavioral_patterns()
        
        report["interesting_games"] = self._identify_interesting_games()
        
        return report
    
    def _analyze_behavioral_patterns(self) -> Dict[str, Any]:
        """Analyze emergent behavioral patterns"""
        
        patterns = {
            "deception_strategies": [],
            "voting_coalitions": [],
            "discussion_patterns": {}
        }
        
        killer_messages = []
        for game in self.game_metrics:
            for player, role in game.player_roles.items():
                if role == "killer" and player in game.discussion_quality:
                    killer_messages.append({
                        "player": player,
                        "model": game.player_models[player],
                        "metrics": game.discussion_quality[player]
                    })
        
        model_deception = defaultdict(list)
        for msg in killer_messages:
            model_deception[msg["model"]].append(msg["metrics"])
        
        for model, metrics_list in model_deception.items():
            if metrics_list:
                patterns["deception_strategies"].append({
                    "model": model,
                    "avg_deception_score": np.mean([m.get("deception_attempts", 0) for m in metrics_list]),
                    "avg_blending_score": np.mean([m.get("blending_score", 0) for m in metrics_list])
                })
        
        return patterns
    
    def _identify_interesting_games(self) -> List[Dict[str, Any]]:
        """Identify games with interesting patterns"""
        
        interesting = []
        
        for game in self.game_metrics[-10:]:  
            interest_score = 0
            reasons = []
            
            if game.rounds > np.mean([g.rounds for g in self.game_metrics]) + np.std([g.rounds for g in self.game_metrics]):
                interest_score += 2
                reasons.append("unusually_long_game")
            
            for voting in game.voting_patterns:
                if len(voting.get("voting_blocks", [])) > 0:
                    interest_score += 1
                    reasons.append("voting_coalitions_formed")
                    break
            
            total_messages = sum(dq.get("total_messages", 0) for dq in game.discussion_quality.values())
            if total_messages > 50:
                interest_score += 1
                reasons.append("high_discussion_activity")
            
            if interest_score >= 2:
                interesting.append({
                    "game_id": game.game_id,
                    "interest_score": interest_score,
                    "reasons": reasons,
                    "rounds": game.rounds,
                    "winner": game.winner
                })
        
        return sorted(interesting, key=lambda x: x["interest_score"], reverse=True)[:5]
    
    def save_results(self):
        """Save evaluation results to disk"""
        
        with open(self.output_dir / "game_metrics.pkl", "wb") as f:
            pickle.dump(self.game_metrics, f)
        
        with open(self.output_dir / "player_metrics.pkl", "wb") as f:
            pickle.dump(self.player_metrics, f)
        
        report = self.generate_report(len(self.game_metrics))
        with open(self.output_dir / "evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        self._generate_visualizations()
    
    def _generate_visualizations(self):
        """Generate charts and visualizations"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        models = list(self.model_comparisons.keys())
        win_rates = [self.model_comparisons[m]["wins"] / self.model_comparisons[m]["games"] 
                    for m in models]
        
        axes[0, 0].bar(models, win_rates)
        axes[0, 0].set_title("Overall Win Rate by Model")
        axes[0, 0].set_ylabel("Win Rate")
        axes[0, 0].set_ylim(0, 1)
        
        killer_rates = [self.model_comparisons[m]["wins_as_killer"] / max(self.model_comparisons[m]["games"], 1)
                       for m in models]
        innocent_rates = [self.model_comparisons[m]["wins_as_innocent"] / max(self.model_comparisons[m]["games"], 1)
                         for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, killer_rates, width, label='As Killer')
        axes[0, 1].bar(x + width/2, innocent_rates, width, label='As Innocent')
        axes[0, 1].set_title("Win Rate by Role")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)
        
        game_lengths = [gm.rounds for gm in self.game_metrics]
        axes[1, 0].hist(game_lengths, bins=10, edgecolor='black')
        axes[1, 0].set_title("Game Length Distribution")
        axes[1, 0].set_xlabel("Number of Rounds")
        axes[1, 0].set_ylabel("Frequency")
        
        winner_counts = Counter(gm.winner for gm in self.game_metrics)
        axes[1, 1].pie(winner_counts.values(), labels=winner_counts.keys(), autopct='%1.1f%%')
        axes[1, 1].set_title("Win Distribution")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_overview.png", dpi=300)
        plt.close()
        
        logger.info(f"Visualizations saved to {self.output_dir}")

class MafiaBenchmarkRunner:
    """Main benchmark runner that orchestrates everything"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.game_engine = None
        self.evaluator = ModelEvaluator()
        self.model_configs = load_model_configs()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load benchmark configuration from file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        self.num_games = config.get("num_games", 100)
        self.game_settings = config.get("game_settings", {})
        self.model_assignments = config.get("model_assignments", {})
    
    async def run_single_game(self, game_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single game with specified configuration"""
        
        engine = MafiaGameEngine(
            num_players=game_config.get("num_players", 6),
            num_killers=game_config.get("num_killers", 2),
            enable_special_roles=game_config.get("enable_special_roles", False)
        )
        
        engine.initialize_game()
        
        player_models = {}
        available_models = list(self.model_configs.keys())
        
        for player_id in engine.game_state.players.keys():
            if player_id in game_config.get("model_assignments", {}):
                model_name = game_config["model_assignments"][player_id]
            else:
                model_name = available_models[hash(player_id) % len(available_models)]
            
            if model_name in self.model_configs:
                adapter = create_model_adapter(model_name, self.model_configs[model_name])
                engine.register_model_adapter(player_id, adapter)
                player_models[player_id] = model_name
            else:
                logger.warning(f"Model {model_name} not configured, using default")
                adapter = create_model_adapter("gpt-3.5", self.model_configs.get("gpt-3.5"))
                engine.register_model_adapter(player_id, adapter)
                player_models[player_id] = "gpt-3.5"
        
        logger.info(f"Starting game with players: {player_models}")
        game_result = await engine.run_game()
        
        self.evaluator.process_game(game_result, player_models, engine.game_log)
        
        return game_result
    
    async def run_benchmark(self, num_games: int = 100, 
                          parallel_games: int = 5) -> Dict[str, Any]:
        """Run full benchmark with multiple games"""
        
        logger.info(f"Starting benchmark with {num_games} games")
        
        game_configs = []
        for i in range(num_games):
            config = {
                "num_players": 6 if i % 3 == 0 else 8,  
                "num_killers": 2 if i % 2 == 0 else 1,  
                "enable_special_roles": i % 4 == 0,  
                "game_id": f"game_{i:04d}"
            }
            game_configs.append(config)
        
        for i in range(0, num_games, parallel_games):
            batch = game_configs[i:i + parallel_games]
            
            tasks = [self.run_single_game(config) for config in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            completed = i + len(batch)
            logger.info(f"Progress: {completed}/{num_games} games completed")
            
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Game {i+j} failed: {result}")
        
        final_report = self.evaluator.generate_report(num_games)
        
        self.evaluator.save_results()
        
        logger.info("Benchmark completed!")
        logger.info(f"Results saved to: {self.evaluator.output_dir}")
        
        return final_report
    
    def analyze_specific_game(self, game_id: str) -> Dict[str, Any]:
        """Deep dive into a specific game"""
        
        game = next((g for g in self.evaluator.game_metrics if g.game_id == game_id), None)
        
        if not game:
            return {"error": f"Game {game_id} not found"}
        
        analysis = {
            "game_id": game_id,
            "basic_info": {
                "rounds": game.rounds,
                "winner": game.winner,
                "players": game.player_models,
                "roles": game.player_roles,
                "survivors": game.final_survivors
            },
            "discussion_analysis": game.discussion_quality,
            "voting_analysis": game.voting_patterns,
            "key_moments": self._identify_key_moments(game)
        }
        
        return analysis
    
    def _identify_key_moments(self, game: GameMetrics) -> List[Dict[str, Any]]:
        """Identify key turning points in a game"""
        
        moments = []
        
        for i, voting in enumerate(game.voting_patterns):
            eliminated_player = None  
            
            vote_dist = voting.get("vote_distribution", {})
            if vote_dist:
                max_votes = max(vote_dist.values())
                close_candidates = [p for p, v in vote_dist.items() if v >= max_votes - 1]
                
                if len(close_candidates) > 1:
                    moments.append({
                        "round": i + 1,
                        "type": "close_vote",
                        "details": f"Close vote between {close_candidates}"
                    })
        
        return moments

async def main():
    """Example of running the benchmark"""
    
    runner = MafiaBenchmarkRunner()
    
    test_config = {
        "num_games": 10,
        "game_settings": {
            "num_players": 6,
            "num_killers": 2,
            "enable_special_roles": True
        }
    }
    
    report = await runner.run_benchmark(
        num_games=test_config["num_games"],
        parallel_games=2
    )
    
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Total games: {report['summary']['total_games']}")
    print(f"Average game length: {report['summary']['avg_rounds_per_game']:.1f} rounds")
    print(f"Killer win rate: {report['summary']['killer_win_rate']:.1%}")
    print(f"Innocent win rate: {report['summary']['innocent_win_rate']:.1%}")
    
    print("\n=== MODEL PERFORMANCE ===")
    for model, perf in report['model_performance'].items():
        print(f"\n{model}:")
        print(f"  Overall win rate: {perf['win_rate']:.1%}")
        print(f"  As killer: {perf['killer_performance']:.1%}")
        print(f"  As innocent: {perf['innocent_performance']:.1%}")
        print(f"  Discussion engagement: {perf['avg_discussion_engagement']:.1f} messages/game")
    
    print("\n=== INTERESTING GAMES ===")
    for game in report['interesting_games'][:3]:
        print(f"\nGame {game['game_id']}:")
        print(f"  Interest score: {game['interest_score']}")
        print(f"  Reasons: {', '.join(game['reasons'])}")
        print(f"  Length: {game['rounds']} rounds")
        print(f"  Winner: {game['winner']}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())
