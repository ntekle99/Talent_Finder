from datetime import UTC, datetime

import streamlit as st

from .scoring import rank_companies

st.set_page_config(page_title="Talent Signal", layout="wide")
st.title("Talent Signal Research")
st.caption("Public-data-first rankings for research and paper trading only.")
st.info("Load normalized TalentEvent objects through a pipeline or notebook. No live orders are supported.")
st.write("The dashboard layer is intentionally thin: rankings, evidence timelines, backtests, and paper positions should be rendered from persisted pipeline outputs.")

if "events" in st.session_state:
    ranked = rank_companies(st.session_state.events, datetime.now(UTC).date())
    st.dataframe(
        [
            {
                "ticker": item.company.ticker,
                "company": item.company.name,
                "score": round(item.score, 3),
                "events": item.event_count,
                "explanation": " | ".join(item.explanations),
            }
            for item in ranked
        ],
        use_container_width=True,
    )
