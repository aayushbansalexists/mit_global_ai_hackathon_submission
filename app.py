import os
import re
import time
import tempfile
import pandas as pd
import numpy as np
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
import PyPDF2
import base64
from dotenv import load_dotenv
from streamlit.components.v1 import html

from google import genai

load_dotenv()

def get_api_key():
    if 'user_api_key' in st.session_state and st.session_state['user_api_key']:
        return st.session_state['user_api_key']
    return os.environ.get("GOOGLE_API_KEY", "")

def ai_chat(messages, model="gemini-2.0-flash", api_key=None):
    if api_key is None:
        api_key = get_api_key()
    client = genai.Client(api_key=api_key)
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text

class DocumentChunk:
    def __init__(self, text, source, page_number):
        self.text = text
        self.source = source
        self.page_number = page_number

class FaissVectorStore:
    def __init__(self):
        self.embedding_dim = 384
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.metadata = []
        
    def add_documents(self, chunks):
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        embs = self.model.encode(texts, convert_to_numpy=True)
        faiss.normalize_L2(embs)
        self.index.add(embs)
        self.metadata.extend(chunks)
        
    def query(self, query_text, top_k=5):
        query_emb = self.model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(distances[0][i])))
        return results

def extract_pdf_chunks(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if not para.strip():
                continue
            chunks.append(DocumentChunk(para, pdf_path, i+1))
    return chunks

def extract_excel_chunks(excel_path):
    chunks = []
    xl = pd.ExcelFile(excel_path)
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        for idx, row in df.iterrows():
            row_text = " ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            if row_text.strip():
                chunks.append(DocumentChunk(row_text, excel_path, idx+1))
    return chunks

def extract_field_with_pattern(text, patterns):
    for pattern in patterns:
        matches = re.search(pattern, text)
        if matches and len(matches.groups()) > 0:
            return matches.group(1).strip()
    return None

def extract_field_with_ai(field, chunks, max_chunks=5):
    if not chunks:
        return None, 0.0, None, None
    context = ""
    for i, chunk in enumerate(chunks[:max_chunks]):
        context += f"Document chunk {i+1} [Source: {os.path.basename(chunk.source)}, Page: {chunk.page_number}]:\n"
        context += f"{chunk.text}\n\n"
    messages = [
        {
            "role": "system",
            "content": f"You are an assistant for extracting specific information from legal and business documents. Extract the value for the field '{field}' from the provided text chunks. Return ONLY the extracted value, nothing else. If you cannot find the value, return 'NOT_FOUND'."
        },
        {
            "role": "user",
            "content": f"Context documents:\n\n{context}\n\nExtract the value for: {field}"
        }
    ]
    try:
        extracted_value = ai_chat(messages)
        if extracted_value == "NOT_FOUND" or "I don't have enough information" in extracted_value or "not found" in extracted_value.lower():
            return None, 0.0, None, None
        confidence = min(0.8, 0.4 + len(extracted_value) / 100)
        best_chunk = None
        best_page = None
        highest_overlap = 0
        for chunk in chunks:
            if extracted_value.lower() in chunk.text.lower():
                return extracted_value, 0.85, chunk.source, chunk.page_number
            words = set(extracted_value.lower().split())
            chunk_words = set(chunk.text.lower().split())
            overlap = len(words.intersection(chunk_words))
            if overlap > highest_overlap:
                highest_overlap = overlap
                best_chunk = chunk.source
                best_page = chunk.page_number
        if best_chunk:
            confidence = min(0.7, confidence * (highest_overlap / max(1, len(words))))
            return extracted_value, confidence, best_chunk, best_page
        return extracted_value, 0.3, chunks[0].source, chunks[0].page_number
    except Exception as e:
        print(f"Error extracting with AI: {str(e)}")
        return None, 0.0, None, None

def prefill_checklist(checklist_fields, chunks):
    field_patterns = {'agreement type': [
            r'(?i)agreement\s+type\s*:\s*([^\.]+)',
            r'(?i)type\s+of\s+agreement\s*:\s*([^\.]+)',
            r'(?i)this\s+([^\.]+?)\s+agreement',
            r'(?i)(power\s+purchase\s+agreement)',
            r'(?i)(lease\s+agreement)',
            r'(?i)(interconnection\s+agreement)'
        ],
        'effective date': [
            r'(?i)effective\s+date\s*[:;]\s*([^\.]+)',
            r'(?i)commenc(?:es|ing|e)\s+on\s*([^\.]+)',
            r'(?i)dated\s+(?:as\s+of\s+)?this\s*([^\.]+)',
            r'(?i)agreement\s+date\s*:\s*([^\.]+)',
            r'(?i)dated\s+(?:this)?\s*(\d{1,2}(?:st|nd|rd|th)?\s+\w+,?\s+\d{4})',
            r'(?i)effective\s+date.*?[:\s]([^\.]+)'
        ],
        'term length': [
            r'(?i)(?:for|of|a)\s+(?:an\s+)?(?:initial\s+)?(?:period|term)\s+of\s+(\d+\s*(?:year|month|day)s?)',
            r'(?i)term\s*(?:of|:)\s*([^\.]+?years|[^\.]+?months)',
            r'(?i)duration\s*(?:of|:)\s*([^\.]+?years|[^\.]+?months)',
            r'(?i)period\s+of\s+([^\.]+?years|[^\.]+?months)',
            r'(?i)shall\s+remain\s+in\s+(?:force|effect)\s+for\s+([^\.]+)',
            r'(?i)term\s+of\s+(\d+)',
            r'(?i)agreement\s+shall\s+(?:be|remain)\s+(?:effective|valid|in\s+effect)\s+for\s+(?:a\s+period\s+of\s+)?(\d+\s*(?:year|month|day)s?)',
            r'(?i)initial\s+term:?\s+(\d+\s*(?:year|month|day)s?)'
        ],
        'counterparty': [
            r'(?i)between\s+([^\.]+)\s+and\s+([^\.]+)',
            r'(?i)parties\s*:\s*([^\.]+)',
            r'(?i)(?:customer|client|developer|lessee|lessor):\s*([^\.]+)',
            r'(?i)(?:this|the)\s+agreement\s+(?:is\s+)?(?:made|entered\s+into)\s+(?:by\s+and\s+)?between\s+([^,]+),?(?:\s+\("?\w+"?\))?\s+and\s+([^,\.]+)'
        ],
        'property details': [
            r'(?i)property\s+description\s*:\s*([^\.]+)',
            r'(?i)located\s+at\s*([^\.]+)',
            r'(?i)premises\s+at\s*([^\.]+)',
            r'(?i)site\s+address\s*:\s*([^\.]+)',
            r'(?i)property\s+(?:located|situated)\s+at\s*([^\.]+)',
            r'(?i)parcel\s+number\s*:\s*([^\.]+)',
            r'(?i)legal\s+description\s*:\s*([^\.]+)',
            r'(?i)premises.*?described\s+in\s+([^\.]+)'
        ],
        'payment terms': [
            r'(?i)payment\s+terms\s*:\s*([^\.]+)',
            r'(?i)compensation\s*:\s*([^\.]+)',
            r'(?i)price\s*:\s*([^\.]+)',
            r'(?i)fee\s+of\s*([^\.]+)',
            r'(?i)rate\s+of\s*\$\s*(\d+(?:\.\d+)?)',
            r'(?i)price\s+of\s*\$\s*(\d+(?:\.\d+)?)',
            r'(?i)compensation\s+of\s*\$\s*(\d+(?:\.\d+)?)',
            r'(?i)rent:?\s+([^\.]+)'
        ],
        'termination clauses': [
            r'(?i)termination\s*:\s*([^\.]+)',
            r'(?i)cancellation\s*:\s*([^\.]+)',
            r'(?i)agreement\s+may\s+be\s+terminated\s*([^\.]+)',
            r'(?i)either\s+party\s+may\s+terminate\s*([^\.]+)',
            r'(?i)right\s+to\s+terminate\s*([^\.]+)',
            r'(?i)termination\s+for\s+(?:default|cause|convenience)\s*([^\.]+)'
        ]
    }
    results = {}
    for field in checklist_fields:
        field_lower = field.lower()
        patterns = field_patterns.get(field_lower, [])
        value_found = False
        for chunk in chunks:
            value = extract_field_with_pattern(chunk.text, patterns)
            if value:
                results[field] = {
                    "value": value,
                    "confidence": 0.95,
                    "source": chunk.source,
                    "page": chunk.page_number
                }
                value_found = True
                break
        if not value_found:
            value, confidence, source, page = extract_field_with_ai(field, chunks)
            if value:
                results[field] = {
                    "value": value,
                    "confidence": confidence,
                    "source": source,
                    "page": page
                }
                value_found = True
        if not value_found:
            results[field] = {
                "value": "",
                "confidence": 0.0,
                "source": "",
                "page": 0
            }
    return results

def display_pdf(file_path, page_number=1):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f"""
    <iframe src="data:application/pdf;base64,{base64_pdf}#page={page_number}" width="100%" height="600px" type="application/pdf"></iframe>
    """
    return pdf_display

class DataRoomProcessor:
    def __init__(self):
        self.vector_store = FaissVectorStore()
        self.chunks = []
        self.file_count = 0
        self.total_chunks = 0
        self.processed_files = {}
        
    def process_file(self, file_path):
        if file_path.lower().endswith('.pdf'):
            chunks = extract_pdf_chunks(file_path)
            self.chunks.extend(chunks)
            self.vector_store.add_documents(chunks)
            self.file_count += 1
            self.total_chunks += len(chunks)
            self.processed_files[file_path] = os.path.basename(file_path)
            return len(chunks)
        elif file_path.lower().endswith(('.xlsx', '.xls')):
            chunks = extract_excel_chunks(file_path)
            self.chunks.extend(chunks)
            self.vector_store.add_documents(chunks)
            self.file_count += 1
            self.total_chunks += len(chunks)
            self.processed_files[file_path] = os.path.basename(file_path)
            return len(chunks)
        return 0
    
    def prefill_checklist(self, checklist_fields):
        return prefill_checklist(checklist_fields, self.chunks)
    
    def query(self, query_text):
        start_time = time.time()
        results = self.vector_store.query(query_text)
        end_time = time.time()
        query_time = end_time - start_time
        return {
            "query": query_text,
            "results": [
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page_number,
                    "relevance": score
                } for chunk, score in results
            ],
            "time_taken": query_time
        }
    
    def rag_query(self, query_text, max_context_chunks=5):
        results = self.vector_store.query(query_text, top_k=max_context_chunks)
        context = ""
        for i, (chunk, score) in enumerate(results):
            context += f"Document {i+1} [Source: {os.path.basename(chunk.source)}, Page: {chunk.page_number}]:\n"
            context += f"{chunk.text}\n\n"
        messages = [
            {
                "role": "system",
                "content": "You are an assistant for renewable energy developers. Answer questions based ONLY on the provided documents. If the answer isn't in the documents, say 'I don't have enough information to answer this question.' Always cite your sources with specific document numbers, page numbers, and quote relevant text."
            },
            {
                "role": "user",
                "content": f"Context documents:\n\n{context}\n\nQuestion: {query_text}\n\nAnswer the question based only on the provided context. Cite specific documents, page numbers, and quotes to support your answer."
            }
        ]
        start_time = time.time()
        answer = ai_chat(messages)
        end_time = time.time()
        query_time = end_time - start_time
        return {
            "query": query_text,
            "answer": answer,
            "context": results,
            "time_taken": query_time
        }
        
    def get_stats(self):
        return {
            "file_count": self.file_count,
            "total_chunks": self.total_chunks,
            "vector_store_size": len(self.vector_store.metadata)
        }

def streamlit_app():
    st.set_page_config(page_title="Energy Document AI Copilot", layout="wide")
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DataRoomProcessor()
        st.session_state.checklist_populated = False
        st.session_state.custom_checklist_populated = False

    st.sidebar.header("API Key Configuration")
    st.sidebar.text_input(
        "Enter your API key for Query AI (never shared)",
        key="user_api_key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", "")
    )
    
    st.title("Energy Document AI Copilot")
    tabs = st.tabs([
        "Upload Documents", 
        "Checklist Extraction", 
        "Custom Extraction", 
        "Query Documents", 
        "RAG Query"
    ])
    
    with tabs[0]:
        st.header("Upload Documents")
        st.write("Upload PDFs and Excel files to process. You can select multiple files.")
        uploaded_files = st.file_uploader("Choose files", type=["pdf", "xlsx", "xls"], accept_multiple_files=True)
        if uploaded_files:
            process_button = st.button("Process Files")
            if process_button:
                progress_bar = st.progress(0)
                status_text = st.empty()
                for i, file in enumerate(uploaded_files):
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(file.getvalue())
                        tmp_path = tmp_file.name
                    status_text.text(f"Processing file {i+1}/{len(uploaded_files)}: {file.name}")
                    chunks_added = st.session_state.processor.process_file(tmp_path)
                    os.unlink(tmp_path)
                progress_bar.progress(1.0)
                status_text.text("All files processed successfully!")
                stats = st.session_state.processor.get_stats()
                st.write(f"Processed {stats['file_count']} files with a total of {stats['total_chunks']} text chunks.")
    
    with tabs[1]:
        st.header("Standard Checklist Extraction")
        st.write("Extract common fields from your renewable energy documents.")
        standard_fields = [
            'agreement type', 'effective date', 'term length', 'counterparty',
            'property details', 'payment terms', 'termination clauses'
        ]
        if st.button("Extract Standard Fields", key="extract_standard"):
            if st.session_state.processor.total_chunks == 0:
                st.error("Please upload and process documents first!")
            else:
                with st.spinner("Extracting standard fields from documents..."):
                    prefilled = st.session_state.processor.prefill_checklist(standard_fields)
                    st.session_state.checklist_results = prefilled
                    st.session_state.checklist_populated = True
        if st.session_state.checklist_populated:
            st.subheader("Extracted Standard Fields")
            for field, data in st.session_state.checklist_results.items():
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.write(f"{field}:")
                with col2:
                    if data["value"]:
                        st.write(data["value"])
                        confidence_str = f"Confidence: {data['confidence']:.2f}"
                        st.write(confidence_str)
                        if data["source"]:
                            source_text = f"Source: {os.path.basename(data['source'])}, Page: {data['page']}"
                            st.write(source_text)
                    else:
                        st.write("Not found")
                with col3:
                    if data["source"] and os.path.exists(data["source"]) and data["source"].lower().endswith('.pdf'):
                        if st.button(f"View PDF", key=f"std_{field}"):
                            st.session_state[f"show_pdf_std_{field}"] = True
                if data["source"] and os.path.exists(data["source"]) and data["source"].lower().endswith('.pdf'):
                    if f"show_pdf_std_{field}" in st.session_state and st.session_state[f"show_pdf_std_{field}"]:
                        st.write(f"*PDF: {os.path.basename(data['source'])}, Page {data['page']}*")
                        st.markdown(display_pdf(data["source"], data["page"]), unsafe_allow_html=True)
                        if st.button("Close PDF", key=f"close_std_{field}"):
                            st.session_state[f"show_pdf_std_{field}"] = False
                st.divider()
    
    with tabs[2]:
        st.header("Custom Fields Extraction")
        st.write("Extract custom fields from your documents by specifying them below.")
        custom_fields_text = st.text_area("Enter custom fields to extract (one per line)", 
                                         placeholder="project capacity\nproject location\ninterconnection point")
        extract_custom_button = st.button("Extract Custom Fields", key="extract_custom")
        if extract_custom_button:
            if st.session_state.processor.total_chunks == 0:
                st.error("Please upload and process documents first!")
            elif not custom_fields_text:
                st.warning("Please enter at least one custom field to extract")
            else:
                custom_fields = [field.strip() for field in custom_fields_text.split("\n") if field.strip()]
                with st.spinner("Extracting custom fields from documents..."):
                    prefilled = st.session_state.processor.prefill_checklist(custom_fields)
                    st.session_state.custom_checklist_results = prefilled
                    st.session_state.custom_checklist_populated = True
        if 'custom_checklist_populated' in st.session_state and st.session_state.custom_checklist_populated:
            st.subheader("Extracted Custom Fields")
            for field, data in st.session_state.custom_checklist_results.items():
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    st.write(f"{field}:")
                with col2:
                    if data["value"]:
                        st.write(data["value"])
                        confidence_str = f"Confidence: {data['confidence']:.2f}"
                        st.write(confidence_str)
                        if data["source"]:
                            source_text = f"Source: {os.path.basename(data['source'])}, Page: {data['page']}"
                            st.write(source_text)
                    else:
                        st.write("Not found")
                with col3:
                    if data["source"] and os.path.exists(data["source"]) and data["source"].lower().endswith('.pdf'):
                        if st.button(f"View PDF", key=f"custom_{field}"):
                            st.session_state[f"show_pdf_custom_{field}"] = True
                if data["source"] and os.path.exists(data["source"]) and data["source"].lower().endswith('.pdf'):
                    if f"show_pdf_custom_{field}" in st.session_state and st.session_state[f"show_pdf_custom_{field}"]:
                        st.write(f"*PDF: {os.path.basename(data['source'])}, Page {data['page']}*")
                        st.markdown(display_pdf(data["source"], data["page"]), unsafe_allow_html=True)
                        if st.button("Close PDF", key=f"close_custom_{field}"):
                            st.session_state[f"show_pdf_custom_{field}"] = False
                st.divider()
    
    with tabs[3]:
        st.header("Query Documents")
        query = st.text_input("Enter your question about the documents", key="standard_query")
        if query and st.button("Search"):
            if st.session_state.processor.total_chunks == 0:
                st.error("Please upload and process documents first!")
            else:
                with st.spinner("Searching documents..."):
                    results = st.session_state.processor.query(query)
                st.subheader(f"Results for: {query}")
                st.write(f"Retrieved in {results['time_taken']:.2f} seconds")
                for i, result in enumerate(results["results"]):
                    with st.expander(f"Result {i+1} - Relevance: {result['relevance']:.2f}"):
                        st.write(f"*Source:* {os.path.basename(result['source'])}, *Page:* {result['page']}")
                        st.write(result["text"])
                        if os.path.exists(result["source"]) and result["source"].lower().endswith('.pdf'):
                            if st.button(f"View PDF", key=f"query_{i}"):
                                st.session_state[f"show_pdf_query_{i}"] = True
                            if f"show_pdf_query_{i}" in st.session_state and st.session_state[f"show_pdf_query_{i}"]:
                                st.write(f"*PDF: {os.path.basename(result['source'])}, Page {result['page']}*")
                                st.markdown(display_pdf(result["source"], result["page"]), unsafe_allow_html=True)
                                if st.button("Close PDF", key=f"close_query_{i}"):
                                    st.session_state[f"show_pdf_query_{i}"] = False
    
    with tabs[4]:
        st.header("RAG Query with Query AI")
        rag_query = st.text_input("Enter your question for Query AI", key="rag_query")
        if rag_query and st.button("Ask Query AI"):
            if st.session_state.processor.total_chunks == 0:
                st.error("Please upload and process documents first!")
            else:
                with st.spinner("Generating response with Query AI..."):
                    try:
                        rag_response = st.session_state.processor.rag_query(rag_query)
                        st.subheader("Query AI Response")
                        st.markdown(rag_response["answer"])
                        st.write(f"Response time: {rag_response['time_taken']:.2f} seconds")
                        st.subheader("Reference Documents")
                        for i, (chunk, score) in enumerate(rag_response["context"]):
                            with st.expander(f"Reference {i+1} - Relevance: {score:.2f}"):
                                st.write(f"*Source:* {os.path.basename(chunk.source)}, *Page:* {chunk.page_number}")
                                st.write(chunk.text)
                                if os.path.exists(chunk.source) and chunk.source.lower().endswith('.pdf'):
                                    if st.button(f"View PDF", key=f"rag_{i}"):
                                        st.session_state[f"show_pdf_rag_{i}"] = True
                                    if f"show_pdf_rag_{i}" in st.session_state and st.session_state[f"show_pdf_rag_{i}"]:
                                        st.write(f"*PDF: {os.path.basename(chunk.source)}, Page {chunk.page_number}*")
                                        st.markdown(display_pdf(chunk.source, chunk.page_number), unsafe_allow_html=True)
                                        if st.button("Close PDF", key=f"close_rag_{i}"):
                                            st.session_state[f"show_pdf_rag_{i}"] = False
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.info("Make sure your API key is valid and you have access to Query AI.")

if __name__ == "__main__":
    streamlit_app()
