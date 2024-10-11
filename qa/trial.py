import os
import logging
import warnings
from typing import List
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars, Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
import time
from langchain.chains.combine_documents import create_stuff_documents_chain

# Logging configuration
logging.getLogger("neo4j").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)


class Neo4jGraph:
    def __init__(self, uri, username, password, max_retries=5, delay=3, pool_size=50, timeout=60):
        self.uri = uri
        self.username = username
        self.password = password
        self.max_retries = max_retries
        self.delay = delay
        self.pool_size = pool_size
        self.timeout = timeout
        self.driver = self.connect_with_retry()

    def connect_with_retry(self):
        for attempt in range(self.max_retries):
            try:
                driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.username, self.password), 
                    max_connection_pool_size=self.pool_size,  
                    connection_acquisition_timeout=self.timeout
                )
                with driver.session() as session:
                    result = session.run("RETURN 1")
                    print(f"Connected to Neo4j successfully on attempt {attempt + 1}")
                    return driver
            except (ServiceUnavailable, ConnectionError) as e:
                print(f"Connection failed on attempt {attempt + 1}: {e}. Retrying in {self.delay} seconds...")
                time.sleep(self.delay)
        raise RuntimeError("Failed to connect to Neo4j")

    def query(self, cypher_query, parameters=None):
        """
        Execute a Cypher query against the Neo4j database and return the results.
        """
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [record.data() for record in result]


class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
                    "appear in the text",
    )


def initialize_system():
    """
    Function to set environment variables and load necessary models (LLM, Neo4j graph, embeddings, etc.)
    """
    load_dotenv()

    # Fallback to environment variables
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    api_key = os.getenv("GROQ_API_KEY")

    try:
        llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")
    except Exception as e:
        print(f"Error loading LLM: {e}")
        raise RuntimeError("Failed to load the LLM")

    # Neo4j graph setup with retry mechanism
    try:
        graph = Neo4jGraph(uri=uri, username=username, password=password)
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        raise RuntimeError("Failed to connect to Neo4j")

    # Embeddings setup for hybrid search
    embeddings_hf = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_index = Neo4jVector.from_existing_graph(
        embedding=embeddings_hf,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

    return llm, graph, vector_index


def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    return full_text_query


def structured_retriever(graph: Neo4jGraph, question: str, entity_chain) -> str:
    # Assuming entity extraction is being handled with the LLM
    entities = entity_chain.invoke({"question": question}).names
    structured_data = ""

    for entity in entities:
        cypher_query = f"""
        CALL db.index.fulltext.queryNodes('entity', $query, {{limit:2}})
        YIELD node, score
        CALL {{
            WITH node
            MATCH (node)-[r:!MENTIONS]->(neighbor)
            RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
            UNION ALL
            WITH node
            MATCH (node)<-[r:!MENTIONS]-(neighbor)
            RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
        }}
        RETURN output LIMIT 50
        """
        structured_data += "\n".join([el['output'] for el in graph.query(cypher_query, {"query": generate_full_text_query(entity)})])
        
    return structured_data or "No structured data found"


def unstructured_retriever(vector_index, question: str) -> str:
    docs = vector_index.similarity_search(question)
    return "\n".join([doc.page_content for doc in docs]) or "No unstructured data found"


def combined_retriever(graph, vector_index, question, entity_chain):
    structured_data = structured_retriever(graph, question, entity_chain)
    unstructured_data = unstructured_retriever(vector_index, question)

    if not structured_data and not unstructured_data:
        return "No relevant results found."
    
    return f"Structured Data:\n{structured_data}\n\nUnstructured Data:\n{unstructured_data}"


def create_chain(llm, entity_chain):
    def retriever(question):
        return combined_retriever(graph, vector_index, question, entity_chain)

    
    qa_prompt = ChatPromptTemplate.from_template(prompt_template)

    # Create `create_stuff_documents_chain` for combining the retrieved documents
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    return question_answer_chain



def get_qa_response(qa_chain, graph, vector_index, question, entity_chain):
    # Get context using the combined retriever function
    context = combined_retriever(graph, vector_index, question, entity_chain)

    # Ensure the input structure matches your prompt template
    entity_input = {
        "context": context,  # Add the context obtained from combined_retriever
        "input": question    # Provide the question or input here
    }

    # Now invoke the entity_chain with the correct input
    try:
        print(f"Entity Input: {entity_input}")

        entities = entity_chain.invoke(entity_input)
        return entities.names  # # Adjust according to your processing logic
    except KeyError as e:
        print(f"KeyError: {e}. Please check your input structure.")



prompt_template = """
You are a helpful customer service assistant. Use the following context to answer the user's question:
Context: {context}
Question: {input}
Answer:
"""


# Initialize system and chain
# llm, graph, vector_index = initialize_system()
# qa_chain = create_chain(llm, graph, vector_index)
llm, graph, vector_index = initialize_system()
entity_chain = create_stuff_documents_chain(llm, ChatPromptTemplate.from_template(prompt_template))  # Create entity chain with llm and prompt
qa_chain = create_chain(llm, entity_chain)

# Example usage
question = "Cost of Amrit myi Baniya da guldasta?"
response = get_qa_response(qa_chain, graph, vector_index, question, entity_chain)

print(response)