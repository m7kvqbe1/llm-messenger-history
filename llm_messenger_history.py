import os
from langchain.document_loaders import FacebookChatLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
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
    loader = FacebookChatLoader(path=folder_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print("Creating embeddings and building vector store...")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("Vector store created.")

    print("Setting up conversational AI...")
    llm = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

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
            print(f"AI: {answer}")
            chat_history.append((query, answer))

    chat()

if __name__ == "__main__":
    main()
