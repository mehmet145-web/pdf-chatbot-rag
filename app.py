import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF Chatbot")

with st.sidebar:
    st.title("⚙️ Ayarlar")

    if st.button("🗑️ Sohbeti Temizle"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.markdown("### Proje Bilgisi")
    st.write("RAG tabanlı PDF chatbot")

    st.markdown("### Kullanılan Teknolojiler")
    st.write("""
    - LangChain
    - FAISS
    - Groq
    - Streamlit
    - HuggingFace
    """)

st.write("PDF yükle, sonra içeriğine göre soru sor.")


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vectorstore_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)
    for chunk in chunks:
         chunk.page_content = (
             chunk.page_content
             .replace(" ş", "ş")
             .replace(" ğ", "ğ")
             .replace(" ı", "ı")
             .replace(" ö", "ö")
             .replace(" ü", "ü")
             .replace(" ç", "ç")
    )
    chunks = [
         chunk for chunk in chunks
         if "................................................................" not in chunk.page_content
]

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.remove(temp_path)

    return vectorstore


def ask_groq(question, context):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
Sen bir PDF chatbot asistanısın.

SADECE verilen PDF bağlamındaki bilgileri kullan.
ASLA kendi genel bilgini ekleme.
PDF içinde geçmeyen ülke, kişi veya olay ekleme.

Eğer bilgi bağlamda yoksa:
"Bu bilgi PDF içinde bulunamadı."
yaz.

Sorunun ana konusunu anlamaya çalış.
Konu anlatımı yapıyormuş gibi kısa özet çıkar.

PDF BAĞLAMI:
{context}

SORU:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Sen PDF içeriğine göre cevap veren dikkatli bir asistansın."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0
    )

    return response.choices[0].message.content


uploaded_file = st.file_uploader(
    "PDF dosyası yükle",
    type=["pdf"]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is not None:
    st.success("PDF yüklendi.")

    current_file = uploaded_file.name

    if st.session_state.get("current_file") != current_file:
         with st.spinner("PDF işleniyor ve vector database oluşturuluyor..."):
             st.session_state.vectorstore = build_vectorstore_from_pdf(uploaded_file)

         st.session_state.current_file = current_file
         st.session_state.messages = []

         st.success("PDF başarıyla işlendi.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("PDF hakkında soru sor...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user", avatar="🧑"):
            st.write(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Cevap hazırlanıyor..."):
                docs = st.session_state.vectorstore.similarity_search(
                     question,
                     k=8
                )

                context = "\n\n".join([
                     f"Sayfa {doc.metadata.get('page')}: {doc.page_content}"
                        for doc in docs
                ])

                answer = ask_groq(question, context)

                pages = sorted(list(set(
                    doc.metadata.get("page") for doc in docs
                )))

                full_answer = f"{answer}\n\n**Kaynak Sayfalar:** {', '.join([str(page) for page in pages])}"

                st.write(full_answer)

                with st.expander("Kaynak metinleri göster"):
                    for i, doc in enumerate(docs, start=1):
                         st.markdown(f"**Kaynak {i} - Sayfa {doc.metadata.get('page')}**")
                         st.write(doc.page_content[:1000])

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_answer
        })

else:
    st.info("Başlamak için bir PDF yükle.")