import os

import requests
import streamlit as st

st.set_page_config(page_title="BioRAG", page_icon="🧬", layout="wide")
st.title("BioRAG")
st.caption("Multimodal, citation-backed biomedical literature exploration")
st.warning("Research prototype. Do not use for diagnosis or clinical decisions.")

api_url = st.sidebar.text_input(
    "API URL", os.getenv("BIORAG_API_URL", "http://localhost:8000")
)
generation_label = st.sidebar.selectbox(
    "Answer model",
    ["Local Ollama (private)", "Gemini API (cloud)"],
    help="Retrieval remains FAISS + BM25. This selects only the answer/vision model.",
)
generation_provider = "ollama" if generation_label.startswith("Local") else "gemini"
if st.sidebar.button("Clear conversation"):
    st.session_state.pop("conversation", None)
    st.session_state.pop("last_result", None)
    st.session_state.pop("search_error", None)
st.session_state.setdefault("conversation", [])
st.sidebar.subheader("Corpus")
uploads = st.sidebar.file_uploader(
    "Add papers or supplementary files",
    type=["pdf", "txt", "md", "csv", "tsv", "xlsx", "xls", "json"],
    accept_multiple_files=True,
)
if st.sidebar.button("Ingest files", disabled=not uploads):
    try:
        files = [("files", (upload.name, upload.getvalue(), upload.type)) for upload in uploads]
        response = requests.post(f"{api_url}/documents/upload", files=files, timeout=600)
        response.raise_for_status()
        st.sidebar.success(f"Indexed: {response.json()['result']}")
    except requests.RequestException as exc:
        st.sidebar.error(f"Ingestion failed: {exc}")

try:
    metrics = requests.get(f"{api_url}/stats", timeout=15)
    if metrics.ok:
        values = metrics.json()
        columns = st.columns(4)
        columns[0].metric("Papers", values["papers"])
        columns[1].metric("Figures", values["figures"])
        columns[2].metric("Tables", values["tables"])
        columns[3].metric("Chunks", values["chunks"])
except requests.RequestException:
    st.info("Start the FastAPI service to view corpus metrics and search.")

question = st.text_input("Ask about the indexed literature")
top_k = st.slider("Evidence passages", 1, 10, 5)

if st.button("Search", type="primary", disabled=not question.strip()):
    try:
        with st.spinner("Retrieving evidence and generating a grounded answer..."):
            response = requests.post(
                f"{api_url}/ask",
                json={
                    "question": question,
                    "top_k": top_k,
                    "history": st.session_state["conversation"][-6:],
                    "generation_provider": generation_provider,
                },
                timeout=180,
            )
        response.raise_for_status()
        st.session_state["last_result"] = response.json()
        st.session_state["conversation"].append(
            {"question": question, "answer": response.json()["answer"]}
        )
        st.session_state.pop("search_error", None)
    except requests.RequestException as exc:
        st.session_state["search_error"] = f"Could not reach the API: {exc}"

if error := st.session_state.get("search_error"):
    st.error(error)

if result := st.session_state.get("last_result"):
    fallback = next(
        (
            step
            for step in result.get("trace", [])
            if step.get("agent") == "provider_router" and "fell back" in step.get("action", "")
        ),
        None,
    )
    if fallback:
        st.warning(
            "Gemini was unavailable, so this answer was generated with local Ollama. "
            f"Reason: {fallback.get('detail', 'provider error')}."
        )
    st.subheader("Answer")
    st.write(result["answer"])
    st.caption(f"Grounding check: {'passed' if result['grounded'] else 'needs review'}")
    st.subheader("Sources")
    for number, (citation, evidence) in enumerate(
        zip(result["citations"], result["evidence"]), start=1
    ):
        with st.expander(f"[{number}] {citation} · score {evidence['score']}"):
            st.write(evidence["text"])
            st.caption(f"Modality: {evidence['modality']}")
    with st.expander("Agent trace"):
        st.json(result["trace"])

if len(st.session_state["conversation"]) > 1:
    with st.expander("Conversation history"):
        for turn in st.session_state["conversation"][:-1]:
            st.markdown(f"**You:** {turn['question']}")
            st.markdown(f"**BioRAG:** {turn['answer']}")
