import os
import re
from typing import List
from functools import lru_cache
import concurrent.futures
import time
import streamlit as st

from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import S3DirectoryLoader
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

page_bg_img = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stAppViewContainer"] > .main {{
background-image: linear-gradient(to right, #000000,#3c3c50);
opacity: 0.8;

}}
</style>
"""



st.markdown(page_bg_img, unsafe_allow_html=True)


# Load environment variables from Streamlit Secrets
aws_access_key_id = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
groq_api_key = st.secrets["groq"]["GROQ_API_KEY"]

# Define course and subject options
COURSES = {
    "BTech": ["cntext", "Compiler Design", "databasesecurity"],
    "ECE": ["Digital Signal Processing", "Embedded Systems"]
}

@lru_cache(maxsize=32)
def load_documents_from_s3(bucket: str, prefix: str) -> List[Document]:
    loader = S3DirectoryLoader(
        bucket,
        prefix=prefix,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    return loader.load()

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text, flags=re.MULTILINE)
    return text.strip()

def process_document(doc: Document) -> Document:
    cleaned_content = clean_text(doc.page_content)
    return Document(page_content=cleaned_content, metadata=doc.metadata)

def create_document_chunks(documents: List[Document], chunk_size: int = 2000, chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        cleaned_documents = list(executor.map(process_document, documents))
    
    chunks = []
    for doc in cleaned_documents:
        chunks.extend(text_splitter.split_documents([doc]))  # Properly flatten the chunks
    return chunks

def create_vector_store(documents: List[Document], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_documents(documents, embedding_model)

def setup_qa_chain(vector_store: FAISS) -> RetrievalQA:
    model = ChatGroq(
        model_name="mixtral-8x7b-32768",
        groq_api_key=groq_api_key,
        temperature=0
    )

    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer and answer with points and at the end give important keywords.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    return qa_chain

def answer_question(qa_chain: RetrievalQA, question: str) -> str:
    result = qa_chain({"query": question})
    answer = result['result']
    sources = [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]
    return f"Answer: {answer}\n\nSources: {', '.join(set(sources))}"

def get_s3_prefix(course: str, subject: str) -> str:
    course_prefix = course.lower()
    subject_prefix = subject.replace(" ", "_")
    return f"{course_prefix}/{subject_prefix}/"

def main():
    st.set_page_config(page_title="S3 QA Chat App", page_icon="🤖")
    st.header("STUDENTBAE 🤖")

    # Initialize session state
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_course' not in st.session_state:
        st.session_state.current_course = None
    if 'current_subject' not in st.session_state:
        st.session_state.current_subject = None

    # Course and subject selection
    col1, col2 = st.columns(2)
    with col1:
        course = st.selectbox("Select Course", list(COURSES.keys()))
    with col2:
        subject = st.selectbox("Select Subject", COURSES[course])

    # Load button
    if st.button("Load Documents"):
        # Check if course or subject has changed
        if course != st.session_state.current_course or subject != st.session_state.current_subject:
            st.session_state.qa_chain = None
            st.session_state.messages = []
            st.session_state.current_course = course
            st.session_state.current_subject = subject

        # S3 bucket and prefix
        bucket = "studentbae"
        prefix = get_s3_prefix(course, subject)

        # Load documents and set up QA chain
        with st.spinner(f"Loading documents for {course} - {subject} from S3 and setting up QA system..."):
            start_time = time.time()

            # Load documents from S3
            documents = load_documents_from_s3(bucket, prefix)
            st.info(f"Loaded {len(documents)} documents from S3")

            # Create document chunks
            chunks = create_document_chunks(documents)
            st.info(f"Created {len(chunks)} cleaned chunks")

            # Create vector store
            vector_store = create_vector_store(chunks)

            # Set up QA chain
            st.session_state.qa_chain = setup_qa_chain(vector_store)

            end_time = time.time()
            st.success(f"Setup completed in {end_time - start_time:.2f} seconds")

    # Display chat interface only if QA chain is set up
    if st.session_state.qa_chain is not None:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if question := st.chat_input(f"Ask a question about {course} - {subject}"):
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = answer_question(st.session_state.qa_chain, question)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Please select a course and subject, then click 'Load Documents' to start.")

if __name__ == "__main__":
    main()
