from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from load_pdf import load_and_split_pdf


def build_vectorstore():
    print("PDF yükleniyor...")

    chunks = load_and_split_pdf("data/sample.pdf")

    print("Embedding modeli yükleniyor...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("FAISS index oluşturuluyor...")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("vectorstore/faiss_index")

    print("FAISS vectorstore oluşturuldu.")
    print(f"Toplam chunk sayısı: {len(chunks)}")


if __name__ == "__main__":
    build_vectorstore()