import json
import os
import time
from langchain.document_loaders import FacebookChatLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

def main():
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Please set your OpenAI API key in the .env file.")

    username = os.getenv("USERNAME")
    if not username:
        raise ValueError("Please set your USERNAME in the .env file.")
    folder_path = f"./data/{username}/messages/inbox"

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder path {folder_path} does not exist.")

    print("Loading Facebook Messenger data...")
    documents = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                loader = FacebookChatLoader(path=file_path)
                documents.extend(loader.load())
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print("Creating embeddings and building vector store...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    batch_size = 100
    vectorstore = None

    for i in range(0, len(docs), batch_size):
        end_idx = min(i + batch_size, len(docs))
        print(f"Processing batch {i//batch_size + 1} of {len(docs)//batch_size + 1}...")

        batch_docs = docs[i:end_idx]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch_docs, embeddings)
        else:
            batch_vectorstore = FAISS.from_documents(batch_docs, embeddings)
            vectorstore.merge_from(batch_vectorstore)

        if end_idx < len(docs):
            print("Waiting 20 seconds before next batch...")
            time.sleep(20)

    print("Vector store creation completed.")

    print("Setting up conversational AI...")
    llm = ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo",
        openai_api_key=OPENAI_API_KEY
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(),
        return_source_documents=True
    )

    def chat():
        print("Chat with your Facebook Messenger AI (type 'exit' to quit):")
        chat_history = []
        while True:
            query = input("You: ")
            if query.lower() in ('exit', 'quit'):
                print("Exiting chat.")
                break
            if not query.strip():
                continue

            result = qa({"question": query, "chat_history": chat_history})
            answer = result["answer"]
            print(f"\nAI: {answer}\n")

            if "source_documents" in result:
                sources = [doc.page_content[:400] for doc in result["source_documents"][:3]]
                sources_json = json.dumps({"sources": sources}, indent=2)
                print(f"\nSources: {sources_json}")

            chat_history.append((query, answer))

    chat()

if __name__ == "__main__":
    main()
