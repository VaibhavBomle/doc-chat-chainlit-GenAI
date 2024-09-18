import os
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl
from dotenv import load_dotenv

load_dotenv()  

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
print(OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(chunk_sie=1000,chunk_overlap=100)

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for user to upload the file
    while files == None:
        files = await cl.AskFileMessage(
            content = "Please upload a text file to begin!",
            accept = ["text/plain"],
            max_size_mb= 20,
            timeout=180
        ).send()
    
    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()

    with open(file.path, "r",encoding="utf-8") as f:
        text = f.read()

    # Split the text into chaunks
    texts = text_splitter.split_text(text)

    # Create a metadata for each chunk
    metadates = [ {"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a chroma vector store and perform embedding
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_text)(
        texts,embeddings,metadates=metadates
    )
    
    # created buffer memeory to sustain previous conversation
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
        
    )

    # Let the user know the system ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()




