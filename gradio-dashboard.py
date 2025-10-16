import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import signal
import sys
import time
import threading

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

import gradio as gr

load_dotenv()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
for _env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_env, "1")

for _env in ("CHROMA_TELEMETRY", "CHROMADB_TELEMETRY", "SENTRY_DSN"):
    os.environ.setdefault(_env, "")

os.environ.setdefault("CHROMA_DISABLE_TELEMETRY", "True")
os.environ.setdefault("CHROMADB_DISABLE_TELEMETRY", "True")
os.environ.setdefault("CHROMA_ANONYMIZE_TELEMETRY", "False")

from langchain_chroma import Chroma

print("Loading books data...")
try:
    books = pd.read_csv("books_with_emotions.csv")
    print(f"Loaded {len(books)} books from CSV")
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"],
    )
    print("Books data processed successfully")
except Exception as e:
    print(f"Error loading books data: {e}")
    raise

raw_documents = TextLoader("tagged_description.txt").load()
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=800,
    chunk_overlap=80,
)
documents = text_splitter.split_documents(raw_documents)

persist_dir = "chroma_books_fastembed"

print("Initializing FastEmbed model...")
try:
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    print("FastEmbed model initialized successfully")
except Exception as e:
    print(f"Error initializing FastEmbed model: {e}")
    raise

print(f"Checking Chroma DB directory: {persist_dir}")
if os.path.isdir(persist_dir) and os.listdir(persist_dir):
    print(f"Loading existing Chroma DB from '{persist_dir}'")
    try:
        db_books = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        print("Chroma DB loaded successfully")
    except Exception as e:
        print(f"Error loading Chroma DB: {e}")
        raise
else:
    print(f"Building Chroma DB and persisting to '{persist_dir}' (first run)")
    try:
        db_books = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        batch_size = 200
        for start in range(0, len(documents), batch_size):
            end = start + batch_size
            batch = documents[start:end]
            print(f"Ingesting documents {start}..{min(end-1, len(documents)-1)} of {len(documents)}")
            db_books.add_documents(batch)
        print("Chroma DB built successfully")
    except Exception as e:
        print(f"Error building Chroma DB: {e}")
        raise


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = []
    for rec in recs:
        text = rec.page_content or ""
        match = re.search(r"\b(97[89]\d{10})\b", text)
        if match:
            try:
                books_list.append(int(match.group(1)))
            except ValueError:
                continue
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

print("Creating Gradio interface...")
try:
    with gr.Blocks() as dashboard:
        gr.Markdown("# Semantic book recommender")

        with gr.Row():
            user_query = gr.Textbox(label = "Please enter a description of a book:",
                                    placeholder = "e.g., A story about forgiveness")
            category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
            tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
            submit_button = gr.Button("Find recommendations")

        gr.Markdown("## Recommendations")
        output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

        submit_button.click(fn = recommend_books,
                            inputs = [user_query, category_dropdown, tone_dropdown],
                            outputs = output)
    print("Gradio interface created successfully")
except Exception as e:
    print(f"Error creating Gradio interface: {e}")
    raise

server_running = False

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    global server_running
    print(f"\nReceived signal {sig}. Shutting down gracefully...")
    server_running = False
    sys.exit(0)

def run_gradio_server():
    """Run Gradio server in a separate thread"""
    global server_running
    try:
        print("Starting Gradio server in background thread...")
        dashboard.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            inbrowser=False,
            prevent_thread_lock=True,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        print(f"Error in Gradio server thread: {e}")
        server_running = False

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    port_env = os.getenv("GRADIO_SERVER_PORT") or os.getenv("PORT")
    port = int(port_env) if port_env else None
    print(f"Launching Gradio on host 0.0.0.0, port={port or 'auto'}")
    print("Gradio app initialized, starting server...")

    server_thread = threading.Thread(target=run_gradio_server, daemon=True)
    server_thread.start()

    server_running = True

    print("Gradio server started successfully")
    print("Server is now running. Press Ctrl+C to stop.")

    try:
        while server_running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server_running = False

    print("Server stopped.")