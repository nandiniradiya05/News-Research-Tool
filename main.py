import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import pickle
import time
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from  langchain.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

# Access environment variables as if they came from the actual environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = OpenAI(temperature=0.6)

st.title("News Research Tool 📈")
st.sidebar.title("News Article Urls")


urls = []
for i in range(3):
    url = st.sidebar.text_input(f"Url {i+1}")
    urls.append(url)
    
process_url_clicked = st.sidebar.button("Process Urls")
file_path = "faiss_index"
main_placefolder = st.empty()

if process_url_clicked:
    
    #load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placefolder.text("Data Loading started........✅✅✅")
    data = loader.load()
    
    #split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n", "\r\n", "\r", "\t", " "],
        chunk_size=1000,
    )
    main_placefolder.text("Data Splitting started........✅✅✅")
    docs = text_splitter.split_documents(data)
    
    #create embeddings and saveit to faiss index
    embeddings = OpenAIEmbeddings()
    vectorstore_apenai = FAISS.from_documents(docs, embeddings)
    main_placefolder.text("Embeddings creation started........✅✅✅")
    time.sleep(2)
    
    # Save the FAISS index to a directory (replace with your preferred path)
    vectorstore_apenai.save_local(file_path)

    main_placefolder.text("FAISS index saved successfully! ✅✅✅")
    
    
    # vectorstore_apenai = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # main_placefolder.text("FAISS index loaded successfully! ✅✅✅")
  
    # with open("faiss_index.pkl", "wb") as f:
    #     pickle.dump(vectorstore_apenai, f)
    
query = main_placefolder.text_input("Enter your question: ")
if query:
    if os.path.exists(file_path):
        # with open(file_path, "rb") as f:
        #     vectorstore_apenai = pickle.load(f)
        embeddings = OpenAIEmbeddings()
        
        # Load FAISS index
        vectorstore_apenai = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)

        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore_apenai.as_retriever())

        # Run the query through the chain
        result = chain.run(query)
        st.header("Answer:")
        st.write(result)