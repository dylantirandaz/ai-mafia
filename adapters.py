import asyncio
import json
import os
import re
from typing import Dict, List, Any, Optional
import aiohttp
import anthropic
import openai
from google import generativeai as genai
import backoff
from dataclasses import dataclass
import logging
from engine import *
from evals import *

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for different AI models"""
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 500
    system_prompt: Optional[str] = None
    
class ClaudeAdapter(ModelAdapter):
    """Adapter for Claude models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
        self.conversation_history = {}
        
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def get_action(self, 
                        game_state: GameState, 
                        player_id: str, 
                        phase: Phase,
                        legal_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get action from Claude"""
        
        if phase == Phase.NIGHT:
            prompt = PromptBuilder.build_night_action_prompt(
                player_id, 
                game_state.players[player_id].role, 
                game_state
            )
        elif phase == Phase.VOTING:
            prompt = PromptBuilder.build_voting_prompt(
                player_id,
                game_state.players[player_id].role,
                game_state,
                self._get_voting_history(player_id)
            )
        else:
            return {"action": "wait"}
        
        if player_id not in self.conversation_history:
            self.conversation_history[player_id] = [{
                "role": "user",
                "content": PromptBuilder.build_role_prompt(
                    player_id,
                    game_state.players[player_id].role,
                    game_state
                )
            }]
        
        try:
            message = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=self.conversation_history[player_id] + [
                    {"role": "user", "content": prompt}
                ],
                system="You are an AI agent playing a social deduction game. Always respond with valid JSON for game actions."
            )
            
            response_text = message.content[0].text
            
            action = self._extract_json(response_text)
            
            self.conversation_history[player_id].extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response_text}
            ])
            
            if len(self.conversation_history[player_id]) > 20:
                self.conversation_history[player_id] = self.conversation_history[player_id][-20:]
            
            return action
            
        except Exception as e:
            logger.error(f"Claude API error for {player_id}: {e}")
            if phase == Phase.VOTING and legal_actions:
                targets = legal_actions[0].get("targets", [])
                if targets:
                    return {"action": "vote", "target": targets[0]}
            return {"action": "wait"}
    
    async def generate_discussion(self,
                                 game_state: GameState,
                                 player_id: str,
                                 discussion_context: List[Dict[str, Any]]) -> str:
        """Generate discussion message with Claude"""
        
        prompt = PromptBuilder.build_discussion_prompt(
            player_id,
            game_state.players[player_id].role,
            game_state,
            model_type="claude"
        )
        
        try:
            message = await self.client.messages.create(
                model=self.config.model_name,
                max_tokens=150,  
                temperature=0.8,  
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="Generate strategic but natural discussion messages. Be concise and avoid revealing your role directly."
            )
            
            return message.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Claude discussion error for {player_id}: {e}")
            return f"I'm still processing what happened. We need to be careful about our votes."
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from model response"""
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        if "kill" in text.lower():
            target_match = re.search(r'Player\d+', text)
            if target_match:
                return {"action": "kill", "target": target_match.group()}
        elif "vote" in text.lower():
            target_match = re.search(r'Player\d+', text)
            if target_match:
                return {"action": "vote", "target": target_match.group()}
        
        return {"action": "wait"}
    
    def _get_voting_history(self, player_id: str) -> List[Dict[str, str]]:
        """Get voting history for this player"""
        return []

class GPT4Adapter(ModelAdapter):
    """Adapter for GPT-4 models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        openai.api_key = config.api_key
        self.client = openai.AsyncOpenAI(api_key=config.api_key)
        
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def get_action(self, 
                        game_state: GameState, 
                        player_id: str, 
                        phase: Phase,
                        legal_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get action from GPT-4"""
        
        if phase == Phase.NIGHT:
            prompt = PromptBuilder.build_night_action_prompt(
                player_id, 
                game_state.players[player_id].role, 
                game_state
            )
        elif phase == Phase.VOTING:
            prompt = PromptBuilder.build_voting_prompt(
                player_id,
                game_state.players[player_id].role,
                game_state
            )
        else:
            return {"action": "wait"}
        
        messages = [
            {
                "role": "system",
                "content": f"""You are playing a Mafia game as {player_id}. 
Your secret role is {game_state.players[player_id].role.value}.
Always respond with valid JSON for game actions.
Be strategic and play to win for your team."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={ "type": "json_object" }  
            )
            
            response_text = response.choices[0].message.content
            action = json.loads(response_text)
            
            return action
            
        except Exception as e:
            logger.error(f"GPT-4 API error for {player_id}: {e}")
            if phase == Phase.VOTING and legal_actions:
                targets = legal_actions[0].get("targets", [])
                if targets:
                    return {"action": "vote", "target": targets[0]}
            return {"action": "wait"}
    
    async def generate_discussion(self,
                                 game_state: GameState,
                                 player_id: str,
                                 discussion_context: List[Dict[str, Any]]) -> str:
        """Generate discussion message with GPT-4"""
        
        prompt = PromptBuilder.build_discussion_prompt(
            player_id,
            game_state.players[player_id].role,
            game_state,
            model_type="gpt4"
        )
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert Mafia player. Generate strategic, concise messages that advance your win condition without revealing your role."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=0.8,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"GPT-4 discussion error for {player_id}: {e}")
            return "We need to think carefully about who to trust here."

class GeminiAdapter(ModelAdapter):
    """Adapter for Google's Gemini models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model_name)
        
    async def get_action(self, 
                        game_state: GameState, 
                        player_id: str, 
                        phase: Phase,
                        legal_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get action from Gemini"""
        
        if phase == Phase.NIGHT:
            prompt = PromptBuilder.build_night_action_prompt(
                player_id, 
                game_state.players[player_id].role, 
                game_state
            )
        elif phase == Phase.VOTING:
            prompt = PromptBuilder.build_voting_prompt(
                player_id,
                game_state.players[player_id].role,
                game_state
            )
        else:
            return {"action": "wait"}
        
        full_prompt = f"""{prompt}

IMPORTANT: Respond ONLY with valid JSON, no other text."""
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                )
            )
            
            response_text = response.text
            action = self._extract_json(response_text)
            
            return action
            
        except Exception as e:
            logger.error(f"Gemini API error for {player_id}: {e}")
            if phase == Phase.VOTING and legal_actions:
                targets = legal_actions[0].get("targets", [])
                if targets:
                    return {"action": "vote", "target": targets[0]}
            return {"action": "wait"}
    
    async def generate_discussion(self,
                                 game_state: GameState,
                                 player_id: str,
                                 discussion_context: List[Dict[str, Any]]) -> str:
        """Generate discussion message with Gemini"""
        
        prompt = PromptBuilder.build_discussion_prompt(
            player_id,
            game_state.players[player_id].role,
            game_state
        )
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.8,
                    max_output_tokens=100,
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini discussion error for {player_id}: {e}")
            return "Let's focus on finding the killers among us."
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from model response"""
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"action": "wait"}

class LocalLLMAdapter(ModelAdapter):
    """Adapter for local LLMs (via HTTP API like Ollama)"""
    
    def __init__(self, config: ModelConfig, api_url: str = "http://localhost:11434"):
        self.config = config
        self.api_url = api_url
        
    async def get_action(self, 
                        game_state: GameState, 
                        player_id: str, 
                        phase: Phase,
                        legal_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get action from local LLM"""
        
        if phase == Phase.NIGHT:
            prompt = PromptBuilder.build_night_action_prompt(
                player_id, 
                game_state.players[player_id].role, 
                game_state
            )
        elif phase == Phase.VOTING:
            prompt = PromptBuilder.build_voting_prompt(
                player_id,
                game_state.players[player_id].role,
                game_state
            )
        else:
            return {"action": "wait"}
        
        payload = {
            "model": self.config.model_name,
            "prompt": f"{prompt}\n\nRespond with JSON only:",
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        response_text = data.get("response", "")
                        return self._extract_json(response_text)
                    else:
                        logger.error(f"Local LLM error: {response.status}")
                        return {"action": "wait"}
                        
        except Exception as e:
            logger.error(f"Local LLM error for {player_id}: {e}")
            return {"action": "wait"}
    
    async def generate_discussion(self,
                                 game_state: GameState,
                                 player_id: str,
                                 discussion_context: List[Dict[str, Any]]) -> str:
        """Generate discussion with local LLM"""
        
        prompt = PromptBuilder.build_discussion_prompt(
            player_id,
            game_state.players[player_id].role,
            game_state
        )
        
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "temperature": 0.8,
            "max_tokens": 100,
            "stream": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "I need more information.").strip()
                    else:
                        return "Something seems off here."
                        
        except Exception as e:
            logger.error(f"Local LLM discussion error: {e}")
            return "We should be careful with our decisions."
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from model response"""
        json_match = re.search(r'\{[^}]+\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"action": "wait"}

class MixedModelAdapter(ModelAdapter):
    """Adapter that can switch between models for diversity"""
    
    def __init__(self, adapters: List[ModelAdapter], strategy: str = "random"):
        self.adapters = adapters
        self.strategy = strategy
        self.call_count = 0
        
    async def get_action(self, 
                        game_state: GameState, 
                        player_id: str, 
                        phase: Phase,
                        legal_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get action using selected strategy"""
        
        adapter = self._select_adapter()
        return await adapter.get_action(game_state, player_id, phase, legal_actions)
    
    async def generate_discussion(self,
                                 game_state: GameState,
                                 player_id: str,
                                 discussion_context: List[Dict[str, Any]]) -> str:
        """Generate discussion using selected adapter"""
        
        adapter = self._select_adapter()
        return await adapter.generate_discussion(game_state, player_id, discussion_context)
    
    def _select_adapter(self) -> ModelAdapter:
        """Select adapter based on strategy"""
        if self.strategy == "random":
            import random
            return random.choice(self.adapters)
        elif self.strategy == "round_robin":
            adapter = self.adapters[self.call_count % len(self.adapters)]
            self.call_count += 1
            return adapter
        else:
            return self.adapters[0]

def create_model_adapter(model_type: str, config: ModelConfig) -> ModelAdapter:
    """Factory function to create appropriate model adapter"""
    
    model_type = model_type.lower()
    
    if "claude" in model_type:
        return ClaudeAdapter(config)
    elif "gpt" in model_type:
        return GPT4Adapter(config)
    elif "gemini" in model_type:
        return GeminiAdapter(config)
    elif "local" in model_type or "ollama" in model_type:
        return LocalLLMAdapter(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_model_configs() -> Dict[str, ModelConfig]:
    """Load model configurations from environment or config file"""
    
    configs = {}
    
    if claude_key := os.getenv("ANTHROPIC_API_KEY"):
        configs["claude-4-opus"] = ModelConfig(
            model_name="claude-opus-4-20250514",
            api_key=claude_key,
            temperature=0.7,
            max_tokens=500
        )
        configs["claude-4-sonnet"] = ModelConfig(
            model_name="claude-sonnet-4-20250514",
            api_key=claude_key,
            temperature=0.7,
            max_tokens=500
        )
    
    # GPT-4 configuration
    if openai_key := os.getenv("OPENAI_API_KEY"):
        configs["gpt-4"] = ModelConfig(
            model_name="gpt-4.1",
            api_key=openai_key,
            temperature=0.7,
            max_tokens=500
        )
        configs["gpt-3.5"] = ModelConfig(
            model_name="gpt-3.5-turbo",
            api_key=openai_key,
            temperature=0.7,
            max_tokens=500
        )
  

### i am gonna add ts later bc gemini is being a pain in the ass rn and ollama is annoying me  
# Gemini configuration
#    if gemini_key := os.getenv("GOOGLE_API_KEY"):
#        configs["gemini-pro"] = ModelConfig(
#            model_name="models/chat-bison-001",
#            api_key=gemini_key,
#            temperature=0.7,
#            max_tokens=500
#        )
    
# Local model configuration (no API key needed)
#    configs["llama2"] = ModelConfig(
#        model_name="llama2",
#        api_key="",
#        temperature=0.7,
#        max_tokens=500
#    )
#    
    return configs
