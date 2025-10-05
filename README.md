## Semantic Book Recommender

An interactive semantic book recommender with sentiment analysis. It indexes book descriptions into a local vector database and serves recommendations using a Gradio UI. Inspired by a tutorial video I found on Youtube:[YouTube video](https://www.youtube.com/watch?v=Q7mS1VHm3Yw) , but modified to use a free HuggingFace embedding model instead of paid OpenAI embeddings.

### What I learned from this:
- **Environment setting**: Creating a new virtual environment using Conda for easier libraries version control.
- **Data cleaning**: How to extract and clean a usable data from a messy one from the original dataset.
- **NLP + Vector Search**: Semantic retrieval over book descriptions using Sentence Transformers and Chroma.
- **Emotion-aware ranking**: Optional re-ranking by emotions (joy, surprise, anger, fear, sadness).
- **Gradio UI**: Gradio Blocks dashboard with categories, tones, and image gallery results.
- **Reproducibility**: Fully local embeddings (no paid API needed) with vector storage caching in your local machine.

### What had been changed?
The original tutorial uses paid OpenAI embeddings (Which I have tried). This version replaces them with a free, local model from HuggingFace (all-MiniLM-L6-v2).

```python
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
```
It uses the same LangChain API, similar retrieval quality for this use case, easy to swap models from HuggingFace.

---

## Stacks used:
- **UI**: Gradio (`gradio-dashboard.py`)
- **RAG building blocks**: LangChain (`TextLoader`, `CharacterTextSplitter`)
- **Embeddings**: `sentence-transformers` via LangChain (`all-MiniLM-L6-v2`)
- **Vector store**: Chroma with local caching (`chroma_books/`)
- **Data**: `books.csv`
- **Analysis/Prep**: Jupyter notebooks (`data-exploration.ipynb`, `sentiment-analysis.ipynb`, `text-classification.ipynb`, `vector-search.ipynb`)

---

## How it works
1. Load descriptions from `tagged_description.txt`.
2. Build or load a persistent Chroma index under `chroma_books/` with Sentence Transformers embeddings.
3. For each user query, retrieve top-k semantically similar books.
4. Optionally re-rank by selected emotion scores from `books_with_emotions.csv`.
5. Display results in a Gradio gallery with thumbnails and concise captions.

---

## Acknowledgments
- Tutorial inspiration: [YouTube video](https://www.youtube.com/watch?v=Q7mS1VHm3Yw)
- Libraries: LangChain, Chroma, Gradio, Sentence Transformers

---


