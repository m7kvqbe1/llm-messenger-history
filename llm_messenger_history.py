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
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from typing import List, Dict
import tiktoken

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def truncate_docs_to_token_limit(docs: List[Dict], max_tokens: int = 6000) -> List[Dict]:
    """Truncate documents to fit within token limit, leaving room for prompt and response."""
    total_tokens = 0
    truncated_docs = []
    
    for doc in docs:
        doc_tokens = estimate_tokens(doc.page_content)
        if total_tokens + doc_tokens > max_tokens:
            break
        truncated_docs.append(doc)
        total_tokens += doc_tokens
    
    return truncated_docs

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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400
    )
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
        max_tokens=2000  # Reserve tokens for response
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 8,  # Reduced from previous value
            "score_threshold": 0.7,
            "fetch_k": 20  # Fetch more but filter down
        }
    )

    def get_relevant_context(query: str, retriever) -> List[Dict]:
        """Get relevant context while managing token limits."""
        docs = retriever.get_relevant_documents(query)
        return truncate_docs_to_token_limit(docs, max_tokens=6000)  # Leave room for prompt and response

    # Add a custom prompt template
    template = """You are an insightful conversation analyst. Analyze these chat messages in detail.
Focus on relationship dynamics, patterns, and meaningful interactions.

Context from messages: {context}
Chat history: {chat_history}

Question: {question}
Detailed Analysis:"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        combine_docs_chain_kwargs={
            'prompt': prompt,
            'document_prompt': PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            )
        },
        return_source_documents=True,
        verbose=True  # Add this to see what's happening
    )

    def analyze_conversation(query, chat_history=[]):
        """Get deeper analysis with multiple perspectives"""
        max_tokens = 8000
        analyses = []
        
        with get_openai_callback() as cb:
            # Core question
            result = qa({
                "question": query,
                "chat_history": chat_history
            })
            analyses.append(("Main Analysis", result["answer"]))
            print(f"\nTokens used: {cb.total_tokens} (Input: {cb.prompt_tokens}, Output: {cb.completion_tokens})")
            
            # Only do follow-ups if we haven't used too many tokens
            if cb.total_tokens < max_tokens:
                follow_ups = [
                    "What emotions and relationship dynamics are evident?",
                    "What interesting patterns or changes can you identify?",
                    "What shared experiences or inside jokes stand out?"
                ]
                
                for follow_up in follow_ups:
                    result = qa({
                        "question": follow_up,
                        "chat_history": chat_history + analyses
                    })
                    analyses.append((follow_up, result["answer"]))
        
        return analyses

    def chat():
        print("Chat with your Facebook Messenger AI (type 'exit' to quit):")
        chat_history = []
        
        while True:
            query = input("\nYou: ")
            if query.lower() in ('exit', 'quit'):
                break
            
            try:
                with get_openai_callback() as cb:
                    result = qa({"question": query, "chat_history": chat_history})
                    print(f"\nAI: {result['answer']}")
                    print(f"\nTokens used: {cb.total_tokens} (Input: {cb.prompt_tokens}, Output: {cb.completion_tokens})")
                    
                    if "source_documents" in result:
                        print("\nBased on these conversations:")
                        for i, doc in enumerate(result["source_documents"][:5], 1):
                            print(f"\n{i}. {doc.page_content[:300]}...")
                    
                    chat_history.append((query, result["answer"]))
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Trying again with reduced context...")
                # Could add retry logic here with reduced context

    chat()

if __name__ == "__main__":
    main()
