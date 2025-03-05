import json
import os
from pathlib import Path
import logging
from neo4j import GraphDatabase

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Neo4jLoader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def load_document(self, doc_data):
        """Load a document into Neo4j from JSON data"""
        with self.driver.session() as session:
            # Extract metadata
            metadata = doc_data.get('metadata', {})
            doc_id = metadata.get('id', os.path.basename(doc_data.get('file_path', 'unknown')))
            title = metadata.get('title', 'Unknown Title')
            summary = doc_data.get('summary', '')
            
            # Create document node
            session.run(
                """
                MERGE (d:Document {id: $doc_id})
                SET d.title = $title,
                    d.summary = $summary
                """,
                doc_id=doc_id,
                title=title,
                summary=summary
            )
            
            # Add entities
            for entity in doc_data.get('entities', []):
                session.run(
                    """
                    MERGE (e:Entity {text: $text})
                    ON CREATE SET e.label = $label, 
                                 e.confidence = $confidence
                    WITH e
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:CONTAINS_ENTITY]->(e)
                    """,
                    text=entity.get('text', ''),
                    label=entity.get('label', 'UNKNOWN'),
                    confidence=entity.get('confidence', 0.5),
                    doc_id=doc_id
                )
            
            # Add topics
            for topic_data in doc_data.get('topics', []):
                topic_id = topic_data.get('id', -1)
                keywords = topic_data.get('keywords', [])
                weight = topic_data.get('weight', 1.0)
                
                session.run(
                    """
                    MERGE (t:Topic {id: $topic_id})
                    SET t.keywords = $keywords, 
                        t.weight = $weight
                    WITH t
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:HAS_TOPIC]->(t)
                    """,
                    topic_id=topic_id,
                    keywords=keywords,
                    weight=weight,
                    doc_id=doc_id
                )
            
            # Add claims
            for claim_data in doc_data.get('claims', []):
                session.run(
                    """
                    MERGE (c:Claim {text: $text})
                    SET c.confidence = $confidence,
                        c.is_hedged = $is_hedged
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:MAKES_CLAIM]->(c)
                    """,
                    text=claim_data.get('text', ''),
                    confidence=claim_data.get('confidence', 0.5),
                    is_hedged=claim_data.get('is_hedged', False),
                    doc_id=doc_id
                )
            
            # Return document ID for confirmation
            return doc_id
            
def load_all_documents(uri, user, password, output_dir):
    """Load all processed documents into Neo4j"""
    output_path = Path(output_dir)
    processed_files = list(output_path.glob("*.json"))
    
    logging.info(f"Found {len(processed_files)} processed documents")
    
    loader = Neo4jLoader(uri, user, password)
    loaded_count = 0
    
    for file in processed_files:
        try:
            with open(file, 'r') as f:
                doc_data = json.load(f)
            
            doc_id = loader.load_document(doc_data)
            logging.info(f"Loaded document {file.name} with ID {doc_id}")
            loaded_count += 1
            
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
    
    loader.close()
    return loaded_count

if __name__ == "__main__":
    # Load configuration
    with open("config.json", 'r') as f:
        config = json.load(f)
    
    neo4j_config = config["neo4j"]
    output_dir = config.get("output_dir", "./output")
    
    # Load all documents
    loaded_count = load_all_documents(
        neo4j_config["uri"], 
        neo4j_config["user"], 
        neo4j_config["password"],
        output_dir
    )
    
    logging.info(f"Successfully loaded {loaded_count} documents into the knowledge graph")