import os
import logging
import warnings
from typing import List
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from pydantic import BaseModel, Field
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars, Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import streamlit as st

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
            except (exceptions.ServiceUnavailable, exceptions.ConnectionError) as e:
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




# Entity extraction model using Pydantic and LLM
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
    # Load environment variables
    load_dotenv()

    try:
        # Local development using secrets.toml or environment variables
        api_key = st.secrets["GROQ_API_KEY"]
        uri = st.secrets["NEO4J_URI"]
        username = st.secrets["NEO4J_USERNAME"]
        password = st.secrets["NEO4J_PASSWORD"]
        print("Secrets loaded from Streamlit.")

    except KeyError:
        # Fallback to environment variables
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME")
        password = os.getenv("NEO4J_PASSWORD")
        api_key = os.getenv("GROQ_API_KEY")
        print("Environment variables loaded.")

    # Load the LLM
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
    try:
        embeddings_hf = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_index = Neo4jVector.from_existing_graph(
            embedding=embeddings_hf,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
    except Exception as e:
        print(f"Error setting up embeddings: {e}")
        raise RuntimeError("Failed to setup embeddings and vector index")

    return llm, graph, vector_index


def create_chain(llm, graph, vector_index):
    """
    Create and return the QA chain using LLM, Neo4j graph, and vector index.
    """
    # Entity extraction prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise and polite customer service bot. ",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ]
    )
    entity_chain = prompt | llm.with_structured_output(Entities)

    def generate_full_text_query(input: str) -> str:
        full_text_query = ""
        words = [el for el in remove_lucene_chars(input).split() if el]
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
        return full_text_query.strip()

    # Define internal retrievers and their functionality
    def structured_retriever(question: str) -> str:
        result = ""
        entities = entity_chain.invoke({"question": question})
        for entity in entities.names:
            response = graph.query(
                """
                CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node, score
                CALL {
                WITH node
                MATCH (node)-[r:!MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:!MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result

    def retriever(question: str):
        structured_data = structured_retriever(question)
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]

        if not structured_data and not unstructured_data:
            # If no data is found, redirect to customer service on WhatsApp
            return "No relevant results found. Please reach out to our customer service on WhatsApp: [Contact Support](https://api.whatsapp.com/send/?phone=9105575000)"
        final_data = f"""Structured data:
        {structured_data}
        Unstructured data:
        {"#Document ". join(unstructured_data)}
            """
        return final_data

    # Condense question prompt for context-based QA
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


    # Runnable branch to determine if the question includes chat history
    _search_query = RunnableBranch(
        # If input includes chat_history, we condense it with the follow-up question
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),  # Condense follow-up question and chat into a standalone_question
            RunnablePassthrough.assign(
                chat_history=lambda x: _format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser(),
        ),
        # Else, we have no chat history, so just pass through the question
        RunnableLambda(lambda x : x["question"]),
    )

    # Final response template
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    # Creating a runnable chain for processing QA responses
    chain = (
        RunnableParallel({"context": _search_query | retriever, "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def get_qa_response(chain, question: str) -> str:
    """
    Function to get an answer for the user's query using the pre-defined QA chain.
    """
    print(f"Input question: {question}")

    # Handle QA process
    try:
        result = chain.invoke({"question": question})
        if not result:
            return (
                "**No relevant results found.**\n\n"
                "Please reach out to our customer service on WhatsApp: "
                "[Contact Support](https://api.whatsapp.com/send/?phone=9105575000)"
            )
        return f"**Response:**\n\n{result}"

    except ServiceUnavailable:
        return (
            "**Sorry, the database is currently unavailable.**\n\n"
            "Please try again later. For immediate assistance, please reach out to our customer service on WhatsApp: "
            "[Contact Support](https://api.whatsapp.com/send/?phone=9105575000)"
        )

    except Exception as e:
        return (
            f"**An error occurred:** `{str(e)}`\n\n"
            "For assistance, please reach out to our customer service on WhatsApp: "
            "[Contact Support](https://api.whatsapp.com/send/?phone=9105575000)"
        )
