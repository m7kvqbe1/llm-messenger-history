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
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Facebook Messenger Chat Analysis')
    parser.add_argument('--model', 
                       choices=['gpt-3.5-turbo', 'gpt-4'],
                       default='gpt-3.5-turbo',
                       help='Choose the OpenAI model to use (default: gpt-3.5-turbo)')
    args = parser.parse_args()

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

    tokens_per_chunk = 250
    rate_limit_tpm = 1000000
    batch_size = 100
    total_chunks = len(docs)
    total_tokens = total_chunks * tokens_per_chunk

    tokens_per_batch = batch_size * tokens_per_chunk
    batches_per_minute = rate_limit_tpm / tokens_per_batch
    optimal_wait_time = (60 / batches_per_minute) * 1.1

    print(f"\nProcessing {total_chunks} chunks (~{total_tokens:,} tokens)")
    print(f"Estimated optimal wait time between batches: {optimal_wait_time:.2f}s")
    print(f"Estimated total processing time: {(total_chunks/batch_size * optimal_wait_time)/60:.1f} minutes\n")

    vectorstore = None
    wait_time = optimal_wait_time
    start_time = datetime.now()

    for i in range(0, len(docs), batch_size):
        end_idx = min(i + batch_size, len(docs))
        batch_num = i//batch_size + 1
        total_batches = len(docs)//batch_size + 1
        elapsed_time = (datetime.now() - start_time).total_seconds() / 60

        print(f"Batch {batch_num}/{total_batches} ({(batch_num/total_batches)*100:.1f}%) - "
              f"Elapsed: {elapsed_time:.1f}m")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                batch_docs = docs[i:end_idx]
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch_docs, embeddings)
                else:
                    batch_vectorstore = FAISS.from_documents(batch_docs, embeddings)
                    vectorstore.merge_from(batch_vectorstore)
                break
            except Exception as e:
                if "Rate limit" in str(e) and attempt < max_retries - 1:
                    wait_time *= 2
                    print(f"Rate limit hit. Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                    wait_time = max(optimal_wait_time, wait_time * 0.75)
                else:
                    raise e

        if end_idx < len(docs):
            print(f"Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)

    total_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"\nVector store creation completed in {total_time:.1f} minutes")

    print(f"Setting up conversational AI using {args.model}...")
    llm = ChatOpenAI(
        temperature=0.7,
        model_name=args.model,
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
