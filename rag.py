import streamlit as st
from neo4j import GraphDatabase
from langchain_openai import OpenAI
import json
import os
from dotenv import load_dotenv
import networkx as nx
from pyvis.network import Network
import tempfile
import base64
from io import BytesIO
from pathlib import Path
import subprocess
import re
import time

# --- Load Configuration ---
# First try to load from .env file
load_dotenv()
api_key_from_env = os.getenv("OPENAI_API_KEY")

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Neo4j RAG Pipeline",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load configuration from config.json
try:
    with open("config.json", "r") as f:
        config = json.load(f)
    neo4j_config = config["neo4j"]
    nlp_config = config["nlp"]

    # Try to get OpenAI API key from config if not found in environment
    if api_key_from_env:
        openai_api_key = api_key_from_env
    elif "openai" in config and "OPENAI_API_KEY" in config["openai"]:
        openai_api_key = config["openai"]["OPENAI_API_KEY"]
    else:
        openai_api_key = None
        st.error("OpenAI API key not found in environment variables or config.json")

except FileNotFoundError:
    st.error("Error: config.json not found. Make sure it's in the same directory.")
    neo4j_config = None
    nlp_config = None
    openai_api_key = api_key_from_env  # Fall back to env var if available
except json.JSONDecodeError:
    st.error("Error: Invalid JSON in config.json.")
    neo4j_config = None
    nlp_config = None
    openai_api_key = api_key_from_env  # Fall back to env var if available
except KeyError as e:
    st.error(f"Error: Missing key in config.json: {e}")
    neo4j_config = None
    nlp_config = None
    openai_api_key = api_key_from_env  # Fall back to env var if available

# 1. Knowledge Graph Connection (Move outside the function for global access)
if neo4j_config:
    uri = neo4j_config["uri"]
    user = neo4j_config["user"]
    password = neo4j_config["password"]

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        # Test connection
        with driver.session() as session:
            session.run("RETURN 1")
        st.success("Connected to Neo4j database")
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {e}")
        driver = None  # Set to None if connection fails
else:
    driver = None
    st.error("Neo4j configuration not loaded. Check config.json.")

# 2. LLM Setup (Move outside the function for global access)
if openai_api_key:
    try:
        llm = OpenAI(openai_api_key=openai_api_key)
        st.success("Connected to OpenAI")
    except Exception as e:
        st.error(f"Failed to initialize OpenAI: {e}")
        llm = None  # Set to None if initialization fails
else:
    llm = None
    st.error("OpenAI API key not provided.")

def generate_query_embedding(query: str):
    """Generate an embedding for the query using OpenAI's embedding API."""
    from openai import OpenAI as OpenAIClient
    try:
        # Initialize the client
        client = OpenAIClient(api_key=openai_api_key)
        
        # Get embeddings
        response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        
        # Extract the embedding from the response
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        st.error(f"Error generating query embedding: {e}")
        return None

def vector_search(query_embedding, documents):
    """Python-side vector similarity calculation"""
    import numpy as np
    from scipy.spatial.distance import cosine
    
    results = []
    
    for doc in documents:
        if doc.get("embedding"):
            # Calculate cosine similarity
            similarity = 1 - cosine(query_embedding, doc["embedding"])
            if similarity > 0.7:  # Same threshold as in Cypher
                results.append({
                    "title": doc["title"],
                    "summary": doc["summary"],
                    "keywords": doc.get("keywords", ""),
                    "similarity": similarity
                })
    
    # Sort by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:5]  # Return top 5

def retrieve_documents(query: str):
    """Retrieve documents from the Neo4j knowledge graph using Python-side vector similarity."""
    if driver is None:
        return [], ""
    
    session = None
    try:
        session = driver.session()
        documents = []
        context = ""
        result_count = 0
        
        # Generate query embedding
        query_embedding = generate_query_embedding(query)
        
        if query_embedding:
            try:
                # Skip APOC attempt and go directly to Python-side vector search
                st.info("Using Python-side vector search")
                
                # Get documents with embeddings (max 50 for performance)
                docs_query = """
                MATCH (d:Document)
                WHERE d.embedding IS NOT NULL
                RETURN d.title as title, d.summary as summary, d.embedding as embedding
                LIMIT 50
                """
                
                results = session.run(docs_query)
                docs = [{
                    "title": r["title"],
                    "summary": r["summary"],
                    "embedding": r["embedding"]
                } for r in results]
                
                # Calculate similarity in Python
                similar_docs = vector_search(query_embedding, docs)
                
                # Process similar docs
                for doc in similar_docs:
                    documents.append({
                        "title": doc["title"], 
                        "summary": doc["summary"], 
                        "keywords": doc.get("keywords", "")
                    })
                    context += f"Document Title: {doc['title']}\nSummary: {doc['summary']}\nSimilarity: {doc['similarity']:.2f}\n\n"
                    result_count += 1
                
            except Exception as e:
                st.warning(f"Vector search failed: {e}. Falling back to keyword search.")
        
        # Rest of your function (keyword search) remains the same
        return documents, context

    except Exception as e:
        return [], f"Error during document retrieval: {e}"

    finally:
        if session:
            session.close()

def generate_answer(context: str, query: str):
    """Generate an answer using the provided context and query."""
    if llm is None:
        return "Error: OpenAI not initialized. Check connection details."

    try:
        # Generate response with context
        augmented_prompt = f"""
        You are an expert scientific research assistant with deep knowledge of scientific literature. Your goal is to answer the user's question as accurately and thoroughly as possible using the provided context from a knowledge graph.

        Here are some guidelines to follow:

        *   **Be concise and to the point.** Avoid unnecessary fluff or introductions.
        *   **Prioritize information from the context.** Only use information that is explicitly provided in the context. Do not rely on external knowledge.
        *   **Synthesize information from multiple sources.** If the answer requires combining information from different documents or topics, do so in a coherent and logical way.
        *   **Cite your sources.** When possible, indicate which document or topic the information is coming from (e.g., "According to Document Title X...").
        *   **If the answer is not found in the context, respond with "I'm sorry, but I cannot find the answer to your question based on the available information."** Do not attempt to answer the question using external knowledge.

        Context:
        {context}

        Question: {query}
        """

        response = llm(augmented_prompt)
        return response

    except Exception as e:
        return f"Error during answer generation: {e}"

def visualize_subgraph(query_text):
    if driver is None:
        return "Neo4j not connected"
    
    try:
        G = nx.Graph()
        
        # Broader visualization query
        cypher_query = """
        MATCH (d:Document)
        WHERE toLower(d.title) CONTAINS toLower($query_param) OR toLower(d.summary) CONTAINS toLower($query_param)
        MATCH (d)-[r]-(n)
        RETURN d, r, n LIMIT 100
        """
        
        with driver.session() as session:
            result = session.run(cypher_query, query_param=query_text)
            
            for record in result:
                doc = record["d"]
                relation = record["r"]
                related_node = record["n"]
                
                # Add document node
                doc_id = f"d_{doc.id}"  # Prefix to avoid ID collisions
                G.add_node(doc_id, label=doc.get("title", "Document"), color="#4287f5", shape="box")
                
                # Add related node
                node_id = f"n_{related_node.id}"  # Prefix to avoid ID collisions
                
                # Color and label based on node type
                if "Topic" in related_node.labels:
                    color = "#f54242"  # Red for topics
                    label = ", ".join(related_node.get("keywords", ["Topic"]))
                elif "Entity" in related_node.labels:
                    color = "#42f5aa"  # Green for entities
                    label = related_node.get("text", "Entity")
                elif "Claim" in related_node.labels:
                    color = "#f5d442"  # Yellow for claims
                    label = related_node.get("text", "Claim")
                else:
                    color = "#a142f5"  # Purple for other nodes
                    label = str(related_node.get("name", "Node"))
                
                G.add_node(node_id, label=label, color=color)
                
                # Add edge
                G.add_edge(doc_id, node_id, title=relation.type)
        
        if len(G.nodes) == 0:
            return "No relevant subgraph found for the query"
            
        net = Network(height="500px", width="100%", notebook=False, bgcolor="#222222", font_color="white")
        net.from_nx(G)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
            net.save_graph(tmp.name)
            return tmp.name
            
    except Exception as e:
        return f"Error visualizing graph: {e}"

def show_graph_stats():
    """Show statistics about the knowledge graph"""
    if driver is None:
        return {"Error": "Neo4j not connected"}
    
    try:
        with driver.session() as session:
            # Get node counts
            result = session.run("""
            CALL {
                MATCH (d:Document) RETURN count(d) as documents
            }
            CALL {
                MATCH (e:Entity) RETURN count(e) as entities
            }
            CALL {
                MATCH (t:Topic) RETURN count(t) as topics
            }
            CALL {
                MATCH (c:Claim) RETURN count(c) as claims
            }
            RETURN documents, entities, topics, claims
            """)
            
            stats = result.single()
            return {
                "Documents": stats["documents"],
                "Entities": stats["entities"],
                "Topics": stats["topics"],
                "Claims": stats["claims"]
            }
    except Exception as e:
        return {"Error": str(e)}

# Add this function for paper processing
def process_uploaded_paper(uploaded_file):
    """Process an uploaded paper and add it to the knowledge graph."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            # Write the uploaded file content to the temp file
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        st.info(f"Processing paper: {uploaded_file.name}")
        
        # Get the path to the Python interpreter in your virtual environment
        python_path = "/Users/abhishekshah/Desktop/technical_paper_extraction/ner/bin/python"
        
        # Run the document processing script
        cmd = [
            python_path,
            "process_document.py",  # Create this script with the processing logic
            "--file", tmp_path,
            "--title", uploaded_file.name
        ]
        
        with st.spinner("Processing document... This may take a few minutes."):
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                st.error(f"Error processing document: {result.stderr}")
                return False
            else:
                # Check for document ID in the output
                match = re.search(r"Document ID: (\d+)", result.stdout)
                doc_id = match.group(1) if match else None
                
                st.success(f"Document processed successfully! ID: {doc_id}")
                return True
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return False
    finally:
        # Clean up the temporary file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except:
            pass

# Add this function to list knowledge base documents
def list_knowledge_base_documents():
    """Retrieve all documents in the knowledge graph with their metadata."""
    if driver is None:
        return []
    
    try:
        with driver.session() as session:
            # Query for documents with their topics and entity counts
            result = session.run("""
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:HAS_TOPIC]->(t:Topic)
            OPTIONAL MATCH (d)-[:CONTAINS_ENTITY]->(e:Entity)
            WITH d, 
                collect(distinct t.keywords) as topics,
                count(distinct e) as entity_count
            RETURN d.title as title, 
                d.summary as summary,
                entity_count,
                topics,
                datetime({epochmillis: timestamp()}) as retrieved_at
            ORDER BY title
            """)
            
            documents = [{
                "title": r["title"],
                "summary": r["summary"] or "No summary available",
                "entity_count": r["entity_count"],
                "topics": [", ".join(topic) for topic in r["topics"] if topic],
                "retrieved_at": r["retrieved_at"]
            } for r in result]
            
            return documents
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []

# Add this function to download documents as CSV
def get_download_link_csv(df):
    """Generate a link to download the dataframe as CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="documents.csv">Download CSV</a>'
    return href

# --- Streamlit App ---
st.title("Neo4j RAG Pipeline")
st.markdown("Query your scientific document knowledge graph")

# Change the tabs to include new functionality
tab1, tab2, tab3, tab4 = st.tabs(["Query RAG", "Graph Explorer", "Upload Papers", "Knowledge Base"])

with tab1:
    st.markdown("### Ask a question about your scientific documents")
    user_query = st.text_input("For example: What are the main topics in the documents?", key="query_input")
    
    # Add pagination controls
    col1, col2 = st.columns([3,1])
    with col2:
        results_per_page = st.selectbox("Results per page:", [5, 10, 20], key="results_per_page")

    # Add this section if you want to display the retrieved documents before the answer
    if st.button("Get Answer", type="primary"):
        if user_query:
            with st.spinner("Searching knowledge graph..."):
                # First retrieve the documents
                documents, context = retrieve_documents(user_query)  # You'd need to split your function
                
                # Display retrieved documents in a collapsible section
                if documents:
                    with st.expander("Retrieved Documents", expanded=False):
                        for doc in documents:
                            st.markdown(f"**{doc['title']}**")
                            st.markdown(doc['summary'])
                            st.markdown(f"*Keywords: {doc['keywords']}*")
                            st.markdown("---")
                
                # Generate the answer
                with st.spinner("Generating answer..."):
                    answer = generate_answer(context, user_query)
                
                st.markdown("### Answer")
                st.markdown(answer)
                
                if answer:
                    col1, col2, col3 = st.columns([1,1,4])
                    with col1:
                        if st.button("👍 Helpful", key="helpful"):
                            # Store feedback
                            st.success("Thanks for your feedback!")
                    with col2:
                        if st.button("👎 Not helpful", key="not_helpful"):
                            # Store feedback
                            st.error("Thanks for your feedback!")
                    
                    with st.expander("Related Questions"):
                        related_questions = [
                            f"What are the applications of {user_query}?",
                            f"How does {user_query} compare to other methods?",
                            f"What are the limitations of {user_query}?",
                        ]
                        for q in related_questions:
                            if st.button(q, key=f"related_{hash(q)}"):
                                st.session_state.user_query = q
                                st.experimental_rerun()
        else:
            st.warning("Please enter a query.")

with tab2:
    st.markdown("### Explore Knowledge Graph")
    viz_query = st.text_input("Enter search term to visualize related nodes:", key="viz_query")
    if st.button("Visualize", key="viz_button"):
        if viz_query:
            with st.spinner("Generating graph visualization..."):
                graph_html = visualize_subgraph(viz_query)
                if graph_html and graph_html.endswith('.html'):
                    with open(graph_html, 'r', encoding='utf-8') as f:
                        html_data = f.read()
                    st.components.v1.html(html_data, height=600)
                else:
                    st.warning(graph_html or "No visualization could be generated")

# Add tab3 for paper uploads
with tab3:
    st.markdown("### Upload Scientific Papers")
    st.write("Upload PDF files to process and add to your knowledge graph.")
    
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files and st.button("Process Papers", key="process_papers"):
        processed_count = 0
        for uploaded_file in uploaded_files:
            with st.expander(f"Processing {uploaded_file.name}", expanded=True):
                success = process_uploaded_paper(uploaded_file)
                if success:
                    processed_count += 1
        
        if processed_count > 0:
            st.success(f"Successfully processed {processed_count} out of {len(uploaded_files)} papers")
            # Refresh the knowledge graph stats
            with st.sidebar:
                with st.expander("Knowledge Graph Stats", expanded=True):
                    st.write("Refreshing stats...")
                    stats = show_graph_stats()
                    for key, value in stats.items():
                        st.metric(key, value)

# Add tab4 for knowledge base browsing
with tab4:
    st.markdown("### Knowledge Base Contents")
    
    if st.button("Refresh Document List", key="refresh_docs"):
        st.session_state.documents = list_knowledge_base_documents()
    
    if 'documents' not in st.session_state:
        with st.spinner("Loading documents..."):
            st.session_state.documents = list_knowledge_base_documents()
    
    # Show document count
    doc_count = len(st.session_state.documents)
    st.write(f"Found {doc_count} documents in the knowledge graph")
    
    # Convert to dataframe for easier display
    import pandas as pd
    if doc_count > 0:
        # Create a simplified dataframe for display
        display_df = pd.DataFrame([{
            "Title": doc["title"],
            "Topics": ", ".join(doc["topics"][:3]) + ("..." if len(doc["topics"]) > 3 else ""),
            "Entities": doc["entity_count"],
            "Has Summary": "Yes" if doc["summary"] != "No summary available" else "No"
        } for doc in st.session_state.documents])
        
        # Add search/filter
        search_term = st.text_input("Filter documents:", key="doc_filter")
        if search_term:
            filtered_df = display_df[display_df.apply(lambda row: 
                search_term.lower() in row['Title'].lower() or 
                search_term.lower() in row['Topics'].lower(), axis=1)]
            st.dataframe(filtered_df)
        else:
            st.dataframe(display_df)
        
        # Add download link
        st.markdown(get_download_link_csv(display_df), unsafe_allow_html=True)
        
        # Document details view
        st.markdown("### Document Details")
        # Let user select a document to view details
        doc_titles = [doc["title"] for doc in st.session_state.documents]
        selected_title = st.selectbox("Select document to view details:", doc_titles)
        
        if selected_title:
            selected_doc = next((doc for doc in st.session_state.documents if doc["title"] == selected_title), None)
            if selected_doc:
                st.markdown(f"## {selected_doc['title']}")
                st.markdown("### Summary")
                st.write(selected_doc['summary'])
                
                st.markdown("### Topics")
                for topic in selected_doc['topics']:
                    st.markdown(f"- {topic}")
                
                # Add button to visualize this document in the graph
                if st.button("Visualize This Document"):
                    with st.spinner("Generating graph visualization..."):
                        graph_html = visualize_subgraph(selected_title)
                        if graph_html and graph_html.endswith('.html'):
                            with open(graph_html, 'r', encoding='utf-8') as f:
                                html_data = f.read()
                            st.components.v1.html(html_data, height=600)
                        else:
                            st.warning(graph_html or "No visualization could be generated")

# Display information about the application
with st.expander("About this application"):
    st.markdown("""
    This application uses a RAG (Retrieval-Augmented Generation) pipeline to answer questions about scientific documents.
    
    1. Your query is used to search a Neo4j knowledge graph containing scientific documents
    2. Relevant documents and topics are retrieved from the knowledge graph
    3. The retrieved information is used to generate an answer using OpenAI's language model
    
    The knowledge graph contains information about documents, their topics, entities, and relationships extracted during document processing.
    """)

# Add this to your tab1 or in a new tab
with st.sidebar:
    with st.expander("Knowledge Graph Stats"):
        stats = show_graph_stats()
        for key, value in stats.items():
            st.metric(key, value)

# Close the Neo4j driver when the app closes (optional)
if driver:
    driver.close()