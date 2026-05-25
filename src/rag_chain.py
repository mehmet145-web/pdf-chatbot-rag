import os
from dotenv import load_dotenv
from groq import Groq

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "vectorstore/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def ask_groq(question: str, context: str):
    api_key = os.getenv("GROQ_API_KEY")

    client = Groq(api_key=api_key)

    prompt = f"""
Sen bir PDF chatbot asistanısın.

SADECE verilen PDF bağlamındaki bilgileri kullan.

ASLA kendi genel bilgini ekleme.
PDF içinde geçmeyen ülke, kişi veya olay ekleme.

Eğer bilgi bağlamda yoksa:
"Bu bilgi PDF içinde bulunamadı."
yaz.

Kısa ve net cevap ver.

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
             "content": "Sen PDF içeriğine göre cevap veren dikkatli bir asistansın. Cevaplarında tarihsel bilgileri abartma, emin olmadığın çıkarımları kesin bilgi gibi yazma. Türkiye gibi özel durumları açıklarken 'fiilen savaşa girmedi' gibi bağlamdaki ayrıntıları belirt."
},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )

    return response.choices[0].message.content


def answer_question(question: str):
    vectorstore = load_vectorstore()

    docs = vectorstore.similarity_search(question, k=10)

    context = "\n\n".join([doc.page_content for doc in docs])

    answer = ask_groq(question, context)

    pages = [doc.metadata.get("page") for doc in docs]

    return answer, pages


if __name__ == "__main__":
    question = input("PDF'e sorun: ")
    answer_question(question)