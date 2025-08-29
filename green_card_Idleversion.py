import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# ✅ Set up LLM
api_key = "AIzaSyAXog-lK5PI060Pm2FvcTHxMHHuJezJRNs"
llm = GoogleGenerativeAI(google_api_key=api_key, model="gemini-2.0-flash")

# ✅ Cache loading & vectorstore
@st.cache_resource
def load_vectorstore():
    loader = CSVLoader(file_path="green_card_faq2.csv", source_column="Question")
    data = loader.load()
    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=api_key, model="models/embedding-001"
    )
    vectorstore = FAISS.from_documents(data, embeddings)
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Prompt template
system_prompt = """You are a helpful assistant for a restaurant.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just be nice and say something else to entertain the user. 

CONTEXT: {context}
QUESTION: {question}"""

PROMPT = PromptTemplate(template=system_prompt, input_variables=["context", "question"])

# ✅ QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

# ✅ Streamlit UI
st.title("🍃 Green Card FAQ Bot (Gemini-2.0-flash)")
question = st.text_input("Ask a question about green cards:")

if question:
    response = qa_chain.invoke({"query": question})
    st.write("### Answer:")
    st.write(response["result"])

    with st.expander("See sources"):
        for doc in response["source_documents"]:
            st.write(f"- {doc.page_content}")
