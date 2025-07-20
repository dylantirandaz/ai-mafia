import pickle, pandas as pd, streamlit as st
from pathlib import Path
from evals import GameMetrics  

@st.cache_data
def load_game_df(pkl_path="benchmark_results/game_metrics.pkl"):
    p = Path(pkl_path)
    with p.open("rb") as f:
        game_metrics: list[GameMetrics] = pickle.load(f)

    rows = []
    for gm in game_metrics:
        discussion_len = sum(
            m.get("total_messages", 0) 
            for m in gm.discussion_quality.values()
        )

        for player, role in gm.player_roles.items():
            won = (
                (role == "killer" and gm.winner == "killers")
                or (role != "killer" and gm.winner == "innocents")
            )
            rows.append({
                "game_id":           gm.game_id,
                "model":             gm.player_models[player],
                "role":              role,
                "win":               int(won),
                "discussion_length": discussion_len
            })

    return pd.DataFrame(rows)

df = load_game_df()

roles  = st.sidebar.multiselect("Role",  df.role.unique(),  default=df.role.unique())
models = st.sidebar.multiselect("Model", df.model.unique(), default=df.model.unique())
df_filt = df[df.role.isin(roles) & df.model.isin(models)]

winrate = df_filt.groupby("model").win.mean().reset_index(name="win_rate")
st.bar_chart(winrate.set_index("model")["win_rate"])

st.scatter_chart(df_filt, x="discussion_length", y="win", color="model")