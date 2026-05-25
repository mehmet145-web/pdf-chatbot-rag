import streamlit as st

from src.rag_chain import answer_question


st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄"
)

st.title("📄 PDF Chatbot")
st.write("PDF içeriğine göre soru sor.")

question = st.text_input("Sorunu yaz:")

if st.button("Cevapla"):
    if question.strip() == "":
        st.warning("Lütfen bir soru yaz.")
    else:
        with st.spinner("Cevap hazırlanıyor..."):
            answer, pages = answer_question(question)

        st.subheader("Cevap")
        st.write(answer)

        st.subheader("Kaynak Sayfalar")
        st.write(", ".join([str(page) for page in pages]))