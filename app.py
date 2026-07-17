import streamlit as st

st.set_page_config(page_title="arXiv Research Assistant", layout="wide")


@st.cache_resource
def get_answer_fn():
    from src.generate import answer
    return answer


answer = get_answer_fn()

st.title("arXiv Research Assistant")
st.caption("Ask questions about transformer and attention mechanism research papers. Answers are grounded in 29 arXiv papers with inline citations.")

question = st.text_input("Your question", placeholder="How does relative position representation work in transformers?")

if question:
    with st.spinner("Retrieving and generating..."):
        result, sources = answer(question)

    st.markdown("### Answer")
    st.write(result)

    if sources:
        st.markdown("### Retrieved sources")
        for paper_id, chunk_index, content, distance in sources:
            with st.expander(f"{paper_id} — chunk {chunk_index} (distance {distance:.4f})"):
                st.text(content)