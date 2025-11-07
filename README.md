📚 RAG-Based PDF Chatbot (LLM-Powered)

This project allows users to upload a PDF file (like notes, reports, or books) and chat with an AI model (LLM) that answers questions directly from the document’s content.
It uses Retrieval-Augmented Generation (RAG) to provide accurate, context-aware answers based on your uploaded file.

🌐 Live Demo: https://rag-based-chatbot325.streamlit.app/

🚀 Project Overview

Have you ever wished you could talk to your documents?
This project brings that to life using RAG (Retrieval-Augmented Generation) — a technique that combines information retrieval with LLM generation.

After uploading a PDF, the app extracts and stores its text in a searchable format. When a user asks a question, the relevant chunks are retrieved and passed to the LLM, which then generates a context-aware, natural answer.

It’s an ideal tool for students, researchers, and professionals who want to quickly understand or search through long documents.

🧠 Tech Stack

Python 🐍

Streamlit – for the web interface

LangChain / OpenAI API – for LLM integration

FAISS / Chroma – for vector-based text retrieval

PyPDF2 / pdfplumber – for extracting text from PDFs

Pandas & NumPy – for data handling and structure

⚙️ How It Works

The user uploads a PDF file through the Streamlit interface.

The text is extracted, split into chunks, and embedded into a vector database.

When a question is asked, the most relevant chunks are retrieved (Retrieval).

The LLM uses this information to generate a precise answer (Generation).

This combination makes responses both accurate and contextually relevant — the essence of RAG systems.

🖥️ Installation & Setup

To run this project locally:

# Clone the repository
git clone https://github.com/Moin36532/rag_based_chatbot

# Navigate into the project folder
cd YOUR_PROJECT_FOLDER_NAME

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py


Then open your browser and go to:
👉 http://localhost:8501/

📸 Live Demo

Try it online without installation!
👉 https://rag-based-chatbot325.streamlit.app/

🤝 Contributing

Contributions, suggestions, and feature ideas are always welcome!
Feel free to fork the repo and submit a pull request.

🧾 License

This project is open source and available under the MIT License
.
