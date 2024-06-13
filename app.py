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
loader = PyPDFLoader('/mnt/data/managing-your-emotions-joyce-meyer.pdf')
doc = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc = text_splitter.split_documents(doc)

# Define the embeddings and vectorstore
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

vectorstore_disk = Chroma(
    persist_directory="./chroma_db_2",
    embedding_function=embeddings
)

retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 2})

# Define the language model
llm = Ollama(model="llama3", temperature=0.5, num_predict=128)

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the prompt
prompt = ChatPromptTemplate.from_template("""Consider yourself as "EMO", a "personal emotional assistant". Your task is to handle the emotions in user text and give them proper guidance based on the context provided. Given a question input, the task of the model is to identify relevant keywords, sentences, phrases in the question and retrieve corresponding answers from the knowledge base. The model should analyze the input question, extract key terms, and search for similar or related questions in the knowledge base. The output should provide the answers associated with the identified keywords or closely related topics. The model should understand the context of the question, identify relevant keywords, phrases and sentences, and retrieve information from the knowledge base based on these keywords. It should be able to handle variations in question phrasing and retrieve accurate answers accordingly with generative answers like a chatbot answers to the user's query. Do not show "relevant keyword fetched" in the answer simply answer the questions in an intelligent manner.

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
st.set_page_config(page_title="EMO - Personal Emotional Assistant", layout="centered")
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
    }
    .chat-box {
        width: 60%;
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
        background-color: #f9f9f9;
    }
    .user-message, .bot-message {
        display: flex;
        align-items: center;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
   .user-message {
    justify-content: flex-end;
    background-color: #000000;
    color: #ffffff; 
}

.bot-message {
    justify-content: flex-start;
    background-color: #000000;
    color: #ffffff; 
}

    .message-text {
        max-width: 80%;
        margin: 0 10px;
    }
    .message-avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background-color: #ccc;
    }
    .input-container {
        width: 60%;
        display: flex;
        align-items: center;
        margin-top: 10px;
    }
    .input-container input {
        width: 100%;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin-right: 10px;
    }
    .input-container button {
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
st.markdown('<h1>EMO - Personal Emotional Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p>Welcome to EMO, your personal healthcare assistant. Ask me anything about your emotional well-being, and I\'ll provide you with guidance and support.</p>', unsafe_allow_html=True)
st.markdown('<div class="chat-box" id="chat-box">', unsafe_allow_html=True)

# Initialize session state to store chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    if message['user']:
        st.markdown(f'<div class="user-message"><div class="message-text">{message["text"]}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message"><div class="message-text">{message["text"]}</div></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # Close chat-box

# Create a form for input
with st.form(key='query_form', clear_on_submit=True):
    user_input = st.text_input("Enter your question here:", key="input_text")
    submit_button = st.form_submit_button(label='Submit')

if submit_button and user_input:
    st.session_state.chat_history.append({'user': True, 'text': user_input})
    with st.spinner("Processing..."):
        bot_response = gen_response(user_input)
        st.session_state.chat_history.append({'user': False, 'text': bot_response})
        st.experimental_rerun()  # Rerun to display the updated chat history
