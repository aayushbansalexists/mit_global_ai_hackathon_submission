# ⚡ Energy Document AI Copilot

MIT AI Global Hackathon Project

A powerful AI assistant for renewable energy developers that processes large volumes of energy documentation, extracts key information, and provides natural language search capabilities with source references.

---

## 🚀 Key Features

- **Bulk Document Processing**: Handles 10,000+ pages or 100+ files simultaneously
- **Natural Language Querying**: Ask questions about your documents in plain English
- **Source-Referenced Answers**: Responses include exact quotes with page numbers and clickable links
- **Checklist Auto-Population**: Extract key contract fields with 86% average confidence
- **Fast Performance**: Sub-500ms response time for document retrieval
- **Gemini AI Integration**: Uses Google's Gemini API for state-of-the-art language processing

---

## 📋 Use Cases

- **Contract Analysis**: Extract key terms from energy agreements
- **Due Diligence**: Quickly verify information across multiple documents
- **Data Extraction**: Auto-populate company forms with extracted values
- **Knowledge Management**: Search across your entire data room instantly

---

## 🛠 Technical Implementation

| Component             | Technology                                                      |
|-----------------------|------------------------------------------------------------------|
| Vector Search         | FAISS vector database for similarity search in R^384 space       |
| Embedding Model       | SentenceTransformer (`all-MiniLM-L6-v2`) for vectorization       |
| LLM Integration       | Google Gemini API for RAG (Retrieval-Augmented Generation)       |
| Document Processing   | PyPDF2 with optimized chunking strategies                        |
| Pattern Matching      | Specialized regex patterns for renewable energy contracts        |
| Frontend              | Streamlit UI with interactive components and PDF viewing         |

---

## 📊 Performance Metrics

- **Field Extraction Confidence**: 86% average across key fields
- **Response Time**: 501ms mean response time
- **Scaling Capability**: Tested with thousands of document chunks

---

## 🚀 Getting Started

### Prerequisites

Install required packages:

```
pip install -r requirements.txt
```

Run: 
```
streamlit run app.py
```
📖 Usage
Upload Documents: Add PDFs and Excel files to the data room

Extract Information: Auto-populate checklist fields from documents

Query Documents: Ask natural language questions about your documents

View Sources: Click on source links to view original documents

🧠 How It Works
The system uses a Retrieval-Augmented Generation (RAG) architecture:

Document Processing: PDFs and Excel files are chunked into semantic units

Vector Embedding: Each chunk is embedded into a high-dimensional vector

Similarity Search: FAISS retrieves the most relevant chunks per query

Contextual Understanding: Gemini API generates answers using retrieved chunks

Source Attribution: Every response links back to the source document

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
