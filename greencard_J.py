import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# ✅ Set up LLM
# Note: Use a valid API key from a secure source, this is a placeholder.
api_key = "AIzaSyAXog-lK5PI060Pm2FvcTHxMHHuJezJRNs"
try:
    llm = GoogleGenerativeAI(google_api_key=api_key, model="gemini-2.0-flash")
except Exception as e:
    st.error(f"Error initializing LLM: {e}")
    st.stop()

# ✅ Cache loading & vectorstore
@st.cache_resource
def load_vectorstore():
    """Loads and caches the FAISS vector store from the CSV file."""
    try:
        loader = CSVLoader(file_path="green_card_faq2.csv", source_column="Question")
        data = loader.load()
        embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=api_key, model="models/embedding-001"
        )
        vectorstore = FAISS.from_documents(data, embeddings)
        return vectorstore
    except FileNotFoundError:
        st.error("Error: The file 'green_card_faq2.csv' was not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data or creating vector store: {e}")
        st.stop()

vectorstore = load_vectorstore()
# Retrieve the top 3 most similar documents.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Corrected Prompt template
# This prompt is now aligned with the green card data.
system_prompt = """You are a helpful assistant providing information about U.S. green cards.
Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just politely state that you do not have the information.
Do not make up any information.

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
#st.set_page_config(page_title="Green Card FAQ Bot", page_icon="🍃")
st.title("🍃 Green Card FAQ Bot (Gemini-2.0-flash)")
st.caption("Ask me a question about U.S. green cards.")

question = st.text_input("Ask a question:", placeholder="e.g., How do I apply for a green card?")

if question:
    with st.spinner("Thinking..."):
        try:
            response = qa_chain.invoke({"query": question})
            st.write("### Answer:")
            st.write(response["result"])

            with st.expander("See sources"):
                for doc in response["source_documents"]:
                    st.write(f"- *Source:* {doc.metadata.get('source', 'N/A')}")
                    st.write(f"  *Content:* {doc.page_content}")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            st.info("Please try a different question or check your API key.")
