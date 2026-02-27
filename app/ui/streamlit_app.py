import streamlit as st
import requests

st.set_page_config(page_title="Local RAG Copilot", layout="wide")
st.title("📄 Local-First Enterprise RAG Copilot")
st.caption("Hybrid Search (BM25 + Embeddings) • Local answer generation • Citations")

api_url = st.text_input("FastAPI URL", "http://localhost:8000")

st.subheader("1) Ingest documents")
st.write("Put PDF/TXT/MD into `data/raw/` then build the index.")

reset = st.checkbox("Reset index (rebuild from scratch)", value=True)
if st.button("Build / Rebuild Index"):
    r = requests.post(f"{api_url}/ingest/build", params={"reset": str(reset).lower()}, timeout=300)
    st.write(r.json())

st.divider()

st.subheader("2) Ask questions")
q = st.text_area("Question", height=100)

if st.button("Ask"):
    resp = requests.post(f"{api_url}/chat", json={"question": q, "top_k": 5}, timeout=300)
    if resp.status_code != 200:
        st.error(resp.text)
    else:
        data = resp.json()
        st.markdown("### Answer")
        st.write(data["answer"])

        st.markdown("### Retrieved evidence")
        for c in data["contexts"]:
            st.markdown(
                f"**Score:** {c['score']:.4f} | **{c['source']}** | page={c.get('page')} | chunk={c['chunk_id']}"
            )
            st.write(c["text"])
            st.divider()