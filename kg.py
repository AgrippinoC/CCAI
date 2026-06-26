import os, datetime, asyncio, time, datetime, uuid
from dotenv import load_dotenv

from neo4j import GraphDatabase
from neo4j_graphrag.llm import MistralAILLM
from neo4j_graphrag.embeddings import MistralAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.retrievers import VectorRetriever

load_dotenv()

neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

llm = MistralAILLM(model_name="mistral-small-latest", model_params={"temperature": 0})
embedder = MistralAIEmbeddings(model="mistral-embed")
splitter = FixedSizeSplitter(chunk_size=1500, chunk_overlap=100)

def ingest_document(file_path: str):
    """Inserisci il PDF nel kg"""
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        neo4j_database=os.getenv("NEO4J_DATABASE"),
        embedder=embedder,
        text_splitter=splitter,
    )
    result = kg_builder.run(file_path=file_path)
    return result

def retrieval_search(query: str, top_k: int = 5):
    retriever = VectorRetriever(
        driver=neo4j_driver,
        index_name="vector",
        embedder=embedder,
        return_properties=["text", "source"]
    )
    return retriever.search(query_text=query, top_k=top_k)

def get_recent_topics() -> str:
    """Prendi i topics usuati nei post recenti, no ripetizoien"""
    cypher = """
        MATCH (p:Post)-[:HAS_TOPIC]->(t:Topic)
        WHERE p.date IS NOT NULL
        RETURN t.name AS topic
        ORDER BY p.date DESC
        LIMIT 30
    """
    with neo4j_driver.session() as session:
        result = session.run(cypher)
        return [r["topic"] for r in result]
    
def get_topic_gaps() -> str:
    """Restituisci i topic meno recenti"""
    cypher = """
        MATCH (t:Topic)
        WHERE NOT (t)<-[:HAS_TOPIC]-(:Post)
        RETURN t.name AS topic
        LIMIT 30
    """
    with neo4j_driver.session() as session:
        result = session.run(cypher)
        return [r["topic"] for r in result]
    
def get_related_topics(topic: str) -> list:
    """Scova topic correlati semanticamente"""
    cypher = """
        MATCH (t:Topic {name:$topic})-[:RELATED_TO]->(r:Topic)
        RETURN r.name AS topic
    """
    with neo4j_driver.session() as session:
        result = session.run(cypher, topic=topic)
        return [r["topic"] for r in result]

def set_related_topics(topic_name: str, threshold: float = 0.95):
    """Collega i topic correlati calcolando la cross-similarity"""
    cypher = """
    MATCH (target:Topic {name: $topic_name})
    MATCH (other:Topic) WHERE other <> target AND other.embedding IS NOT NULL
    
    WITH other, vector.similarity.cosine(target.embedding, other.embedding) AS similarity
    WHERE similarity >= $threshold
    RETURN other.name AS topic, similarity
    ORDER BY similarity DESC
    """
    with neo4j_driver.session() as session:
        result = session.run(cypher, topic_name=topic_name, threshold=threshold)

def suggest_next_topics():
    """Suggerisci topic mancanti"""
    gaps = set(get_topic_gaps())
    suggestions = list(gaps)
    if not suggestions: return ["Legione, Gladiatori, Monete"]
    return suggestions[:10]

def query_graph_plan(week_key: str) -> str:
    """Controlla se è presente un piano editoriale precedente"""
    cypher = """
        MATCH (n:EditorialPlan)
        WHERE n.week_key=$week_key
        RETURN n.content AS content;
    """
    with neo4j_driver.session() as session:
        result = session.run(cypher, week_key=week_key)
        return [r["content"] for r in result]

def query_graph_rag(query_text: str) -> str:
    """Applica il K-RAG; estende le query usando il KG e cerca documenti"""
    try:
        related = get_related_topics(query_text)
        exp_query = query_text
        if related:
            exp_query += " " + " ".join(related)
        search_result = retrieval_search(exp_query)
        if not search_result.items:
            return "Nessun contenuto nel KG."
        kg_result = []

        for i, item in enumerate(search_result.items):
            content = item.content if hasattr(item, "content") else str(item)
            kg_result.append(
                f"""
                    [RISULTATO {i+1}]
                    CONTENT: {content}
                    SOURCE: {getattr(item, 'source', 'unknown')}
                    DOCUMENT: {getattr(item, 'document', 'unknown')}
                """
            )
        return "\n".join(kg_result)
    except Exception as e:
        return f"Errore nel query_graph_rag: {str(e)}"

def update_post_knowledge_graph(post_text: str, topic: str, sources: list, claims: list):
    curr_date = datetime.date.today().isoformat()
    topic_embedding = [float(x) for x in embedder.embed_query(topic)]
    claims_with_embeddings = [{"text": c, "embedding": [float(x) for x in embedder.embed_query(c)]}for c in claims]
    post_id = str(uuid.uuid4())
    set_related_topics(topic)
    time.sleep(3)
    with neo4j_driver.session() as session:
        with session.begin_transaction() as tx:
            
            tx.run("""
                MERGE (p:Post {id: $post_id})
                SET p.text = $post_text, p.date = $curr_date
                MERGE (t:Topic {name: $topic})
                SET t.embedding = $topic_embedding
                MERGE (p)-[:HAS_TOPIC]->(t)
            """, post_id=post_id, post_text=post_text, curr_date=curr_date, topic=topic, topic_embedding=topic_embedding)

            if sources:
                tx.run("""
                    MATCH (p:Post {id: $post_id})
                    UNWIND $sources AS src
                    MERGE (s:Source {name: src})
                    MERGE (p)-[:USES_SOURCE]->(s)
                """, post_id=post_id, sources=sources)

            if claims_with_embeddings:
                tx.run("""
                    MATCH (p:Post {id: $post_id})
                    UNWIND $claims_list AS item
                    MERGE (c:Claim {text: item.text})
                    SET c.embedding = item.embedding
                    MERGE (p)-[:CONTAINS_CLAIM]->(c)
                """, post_id=post_id, claims_list=claims_with_embeddings)

def save_editorial_plan(week_key: str, plan_text: str) -> bool:
    """Salva il piano editoriale settimanale nel Knowledge Graph"""
    curr_date = datetime.date.today().isoformat()
    
    cypher = """
        MERGE (ep:EditorialPlan {week_key: $week_key})
        SET ep.content = $plan_text,
            ep.updated_at = $curr_date
    """
    try:
        with neo4j_driver.session() as session:
            session.run(
                cypher,
                week_key=week_key,
                plan_text=plan_text,
                curr_date=curr_date
            )
        return True
    except Exception as e:
        print(f"Errore salvataggio piano editoriale nel KG: {e}")
        return False

def check_duplicate_claims(new_claims: list, threshold: float) -> bool:
    """
    Verifica se i nuovi claim sono troppo simili a quelli già registrati nel KG.
    Ritorna True se viene trovato almeno un duplicato, altrimenti False.
    """
    cypher = """
    MATCH (c:Claim) WHERE c.embedding IS NOT NULL
    WITH c, vector.similarity.cosine(c.embedding, $new_embedding) AS similarity
    WHERE similarity >= $threshold
    RETURN c.text AS text, similarity
    LIMIT 1
    """
    with neo4j_driver.session() as session:
        for claim in new_claims:
            emb = [float(x) for x in embedder.embed_query(claim)]
            result = session.run(cypher, new_embedding=emb, threshold=threshold)
            record = result.single()
            if record:
                print(f"Claim duplicato: '{record['text']}' (Sim: {record['similarity']:.2f})")
                return True
    return False

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

async def main(pdf_path: str):

    print(f"Avvio ingestione: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    text_list = [page.extract_text() for page in reader.pages if page.extract_text()]
    full_text = "\n\n".join(text_list)
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], 
        chunk_size=1500, 
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(full_text)
    print(f"Totale chunk da processare: {len(chunks)}")
    
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
            
        print(f"Elaborazione chunk {i+1}/{len(chunks)}...")
        try:
            embedding = embedder.embed_query(chunk)
            cypher = """
                CREATE (c:Chunk {id: randomUUID()})
                SET c.text = $text,
                    c.embedding = $embedding,
                    c.source = $source
                """
            
            with neo4j_driver.session() as session:
                session.run(cypher, text=chunk, embedding=embedding, source=pdf_path)
       
            await asyncio.sleep(8)
            
        except Exception as e:
            print(f"Errore sul chunk {i+1}: {e}. Attendi...")
            await asyncio.sleep(15)
            continue

    print("Processo di ingestione completato")

async def test_retrieval():
    print("Test 1: Ricerca Vettoriale")
    query_test = "Emblema della I LEGIONE ITALICA?" 
    try:
        risultati_vettore = retrieval_search(query_test, top_k=2)
        print(f"Trovati {len(risultati_vettore.items)} risultati.")
        for item in risultati_vettore.items:
            print(f"- Contenuto parziale: {item.content[:150]}...")
    except Exception as e:
        print(f"Errore nella ricerca vettoriale: {e}")
        print("Nota: Assicurati che l'indice denominato 'vector' esista su Neo4j.")

    print("\nTest 2: Graph-RAG")
    try:
        risultato_graph_rag = query_graph_rag("Britannia")
        print(risultato_graph_rag)
    except Exception as e:
        print(f"Errore nel query_graph_rag: {e}")

if __name__ == "__main__":
    #asyncio.run(main("./Docs/ABBIGLIAMENTO.pdf"))
    time.sleep(10)
    #asyncio.run(main("./Docs/The-72-Roman-Emperors.pdf"))
    #time.sleep(3)
    #asyncio.run(main("./Docs/legions.pdf"))
    #asyncio.run(main("./Docs/numeristica.pdf"))
    #time.sleep(3)
    #print("FINIT")
    #asyncio.run(test_retrieval())