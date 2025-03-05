from neo4j import GraphDatabase

def check_connection():
    try:
        # Use the same credentials from your config.json
        with GraphDatabase.driver(
            "bolt://localhost:7687", 
            auth=("neo4j", "Sundead123")
        ) as driver:
            with driver.session() as session:
                result = session.run("RETURN 'Connection successful' as message")
                message = result.single()["message"]
                print(message)
                return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

check_connection()

def verify_graph_data():
    """Verify that data has been loaded into the knowledge graph"""
    with GraphDatabase.driver(
        "bolt://localhost:7687", 
        auth=("neo4j", "Sundead123")
    ) as driver:
        with driver.session() as session:
            # Check node counts
            counts = {}
            for label in ["Document", "Entity", "Topic", "Claim", "Citation"]:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                counts[label] = result.single()["count"]
            
            # Check relationship counts
            rel_counts = {}
            for rel_type in ["HAS_TOPIC", "CONTAINS_ENTITY", "MAKES_CLAIM", "RELATES_TO"]:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                rel_counts[rel_type] = result.single()["count"]
                
            return {
                "nodes": counts,
                "relationships": rel_counts
            }

print(verify_graph_data())

def sample_graph_data():
    """Sample some data from the knowledge graph"""
    with GraphDatabase.driver(
        "bolt://localhost:7687", 
        auth=("neo4j", "Sundead123")
    ) as driver:
        with driver.session() as session:
            # Sample documents
            docs = session.run("""
            MATCH (d:Document) 
            RETURN d.title as title, d.summary as summary
            LIMIT 2
            """)
            documents = [{"title": record["title"], "summary": record["summary"]} 
                        for record in docs]
            
            # Sample topics
            topics = session.run("""
            MATCH (t:Topic) 
            RETURN t.keywords as keywords
            LIMIT 3
            """)
            topic_list = [record["keywords"] for record in topics]
            
            # Sample document-topic connections
            doc_topics = session.run("""
            MATCH (d:Document)-[:HAS_TOPIC]->(t:Topic)
            RETURN d.title as doc, t.keywords as topic
            LIMIT 3
            """)
            connections = [{"doc": record["doc"], "topic": record["topic"]} 
                          for record in doc_topics]
            
            return {
                "documents": documents,
                "topics": topic_list,
                "connections": connections
            }

print(sample_graph_data())

def add_entities(self, doc_id, entities):
    with self.driver.session() as session:
        for entity in entities:
            # Create entity node if it doesn't exist
            session.run(
                "MERGE (e:Entity {text: $text}) "
                "ON CREATE SET e.type = $type "
                "WITH e "
                "MATCH (d:Document) WHERE id(d) = $doc_id "
                "MERGE (d)-[:CONTAINS_ENTITY]->(e)",
                text=entity['text'],
                type=entity['type'],
                doc_id=doc_id
            )