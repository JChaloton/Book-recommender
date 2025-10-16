## Semantic Book Recommender
![hippo](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExcjVmdm4yOW92Yjh5MW0wNDU4aWJieXNhNGJ6eTQzZjFjY3Y5MTZrbCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/tXCYey8S6Y4UFx9n4r/giphy.gif)

An interactive semantic book recommender with sentiment analysis. It indexes book descriptions into a local vector database and serves recommendations using a Gradio UI. Inspired by a tutorial video I found on Youtube:[YouTube video](https://www.youtube.com/watch?v=Q7mS1VHm3Yw), but modified to use a free FastEmbed embedding model.

### What I learned from this:
- **Environment setting**: Creating a new virtual environment using Conda for easier libraries version control.
- **Data cleaning**: How to extract and clean a usable data from a messy one from the original dataset.
- **NLP + Vector Search**: Semantic retrieval over book descriptions using FastEmbed + Chroma.
- **Emotion-aware ranking**: Optional re-ranking by emotions (joy, surprise, anger, fear, sadness).
- **Gradio UI**: Gradio Blocks dashboard with categories, tones, and image gallery results.
- **Reproducibility**: Fully local embeddings with vector storage caching in local machine.

### What had been changed?
The original tutorial uses paid OpenAI embeddings (Which I have tried). I initially replaced them with a free, local model from HuggingFace via Sentence Transformers (`all-MiniLM-L6-v2`). I then switched to **FastEmbed** to avoid heavyweight PyTorch/CUDA dependencies.

```python
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
```
FastEmbed runs on CPU via ONNX Runtime, keeps the image small, and requires no GPU. Current model: `BAAI/bge-small-en-v1.5`.

---

## Stacks used:
- **UI**: Gradio (`gradio-dashboard.py`)
- **RAG building blocks**: LangChain (`TextLoader`, `CharacterTextSplitter`)
- **Embeddings**: FastEmbed via LangChain (`BAAI/bge-small-en-v1.5`)
- **Vector store**: Chroma with local caching (`chroma_books_fastembed/`)
- **Data**: `books.csv`
- **Analysis/Prep**: Jupyter notebooks (`data-exploration.ipynb`, `sentiment-analysis.ipynb`, `text-classification.ipynb`, `vector-search.ipynb`)

---

## How it works
1. Load descriptions from `tagged_description.txt`.
2. Build or load a persistent Chroma index under `chroma_books_fastembed/` with FastEmbed embeddings.
3. For each user query, retrieve top-k semantically similar books.
4. Optionally re-rank by selected emotion scores from `books_with_emotions.csv`.
5. Display results in a Gradio gallery with thumbnails and concise captions.

---

## Run locally (Python)
1. Install dependencies from `requirements.txt`.
2. Run:
```bash
python gradio-dashboard.py
```
The app listens on `http://localhost:7860/` by default.

## Run with Docker
Build the image:
```bash
docker build -t book-recommender:latest .
```
Run the container:
```bash
docker run --rm -p 7860:7860 book-recommender:latest
```
The app will be available at `http://localhost:7860/`.

---

## Acknowledgments
- Tutorial inspiration: [YouTube video](https://www.youtube.com/watch?v=Q7mS1VHm3Yw)
- Libraries: LangChain, Chroma, Gradio, FastEmbed

---


