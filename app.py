import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from ollama import embeddings
import pickle
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the document
loader = PyPDFLoader('managing-your-emotions-joyce-meyer.pdf')
doc = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc = text_splitter.split_documents(doc)

# Define the embeddings and vectorstore
embeddings = OllamaEmbeddings(model="mistral")

vectorstore_disk = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 2})

# Define the language model
llm = Ollama(model="mistral", temperature=0.5, num_predict=128)

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the prompt
prompt = ChatPromptTemplate.from_template("""Consider yourself as "EMO", a "personal healthcare assistant". Your task is to handle the emotions in user text and give them proper guidance based on the context provided. Given a question input, the task of the model is to identify relevant keywords, sentences, phrases in the question and retrieve corresponding answers from the knowledge base. The model should analyze the input question, extract key terms, and search for similar or related questions in the knowledge base. The output should provide the answers associated with the identified keywords or closely related topics. The model should understand the context of the question, identify relevant keywords, phrases and sentences, and retrieve information from the knowledge base based on these keywords. It should be able to handle variations in question phrasing and retrieve accurate answers accordingly with generative answers like a chatbot answers to the user's query. Do not show "relevant keyword fetched" in the answer simply answer the questions in an intelligent manner.

Context:
{context}

Question:
{question}""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Function to generate response
def gen_response(question):
    if question.lower() in ["hi", "hi!", "hello", "hello!"]:
        return "Hello! How can I assist you today?"

    answer = rag_chain.invoke(question)
    return answer

# Streamlit UI
st.title("EMO - Personal Healthcare Assistant")
st.write("Welcome to EMO, your personal healthcare assistant. Ask me anything about your emotional well-being, and I'll provide you with guidance and support.")

# Create a form for input
with st.form(key='query_form'):
    question = st.text_input("Enter your question here:")
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    if question:
        with st.spinner("Processing..."):
            response = gen_response(question)
            st.success(response)
    else:
        st.warning("Please enter a question to get a response.")
