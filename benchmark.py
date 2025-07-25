import asyncio
import os
from evals import MafiaBenchmarkRunner
from dotenv import load_dotenv
load_dotenv()   
print("ANTHROPIC_API_KEY:", bool(os.getenv("ANTHROPIC_API_KEY")))
print("OPENAI_API_KEY:  ", bool(os.getenv("OPENAI_API_KEY")))
print("GOOGLE_API_KEY:  ", bool(os.getenv("GOOGLE_API_KEY")))
async def run_simple_benchmark():
    runner = MafiaBenchmarkRunner("config.json")
    print("model_configs keys:", list(runner.model_configs.keys()))

    report = await runner.run_benchmark(num_games=1) # change for diff amt of games, i do not have enough money to run too many but feel free
    
    print(f"Killer win rate: {report['summary']['killer_win_rate']:.1%}")
    print(f"Best performing model: {max(report['model_performance'], key=lambda x: report['model_performance'][x]['win_rate'])}")

asyncio.run(run_simple_benchmark())
