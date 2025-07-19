# AI Is Among Us

An Among Us / Mafia inspired benchmark: watch LLMs (GPT‑4, Claude, more soon…) deceive eachother and work together to figure out who the killer is.

## Quickstart

1. **Clone & install**
   ```bash
   git clone https://dylantirandaz/ai-mafia.git
   python -m venv venv
   source venv/bin/activate  # or  venv\Scripts\activate`
   pip install -r requirements.txt
   ```

2. **Set API keys**
   ```powershell
   $Env:OPENAI_API_KEY="…"
   $Env:ANTHROPIC_API_KEY="…"
   $Env:GOOGLE_API_KEY="…"
   ```

3. **Verify model names** in `adapters.py` (e.g. `"gpt-4.1"`, `"claude-3-opus"`).

4. **Run**
   ```bash
   python benchmark.py
   ```

## Config

Edit **config.json** for  
- `num_games`  
- `players_per_game`  
- `role weights, timeouts, etc.`

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
