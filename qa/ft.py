import os
import logging
import warnings
from typing import List, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars, Neo4jVector
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



# Validate input text to prevent empty or overly long inputs
def validate_input(question: str):
    if not question or len(question.strip()) == 0:
        raise ValueError("The input text is empty.")
    if len(question) > 500:  # Adjust this based on model input limits
        raise ValueError("Input text is too long.")
    return question.strip()


# Extract entities from the text using the LLM
def get_entities_from_text(question: str):
    """Extract entities from the provided text using the LLM."""
    try:
        validated_question = validate_input(question)
        
        # Generate the prompt input
        prompt_input = prompt.format(question=validated_question)
        
        # Invoke the LLM
        response = entity_chain.invoke(prompt_input)  # or however you call your chain
        
        # Process the response
        entities = process_response(response["choices"][0]["message"]["content"])

        if not entities or 'names' not in entities:
            raise ValueError("No entities found in the response.")
        return entities

    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        return {"error": str(ve)}

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTPError: {http_err.response.status_code} - {http_err.response.text}")
        return {"error": "There was an issue processing your request. Please try again later."}

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return {"error": "An unexpected error occurred. Please contact support."}



# Retry entity extraction in case of transient errors
def get_entities_with_retry(question: str, retries=3, delay=2):
    attempt = 0
    while attempt < retries:
        try:
            return get_entities_from_text(question)
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
            time.sleep(delay)
            attempt += 1
    return {"error": "Failed to process the request after multiple attempts."}


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
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "You are a concise and polite customer service bot. ",
    #         ),
    #         (
    #             "human",
    #             "Use the given format to extract information from the following "
    #             "input: {question}",
    #         ),
    #     ]
    # )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a concise and polite assistant, designed to **strictly extract** important entity names such as people, organizations, and businesses from text. **Your response must only include the names without any extra information, context, or links.** Return the information in a structured JSON format."
            ),
            (
                "human",
                "Please strictly extract entity names from the following text. Return a JSON object with the key 'names' as a list of entity names."
            )
        ]
    )

    entity_chain = prompt | llm.with_structured_output(Entities)

    def generate_full_text_query(input: str) -> str:
        words = remove_lucene_chars(input).split()
        return " AND ".join([f"{word}~2" for word in words]) if words else ""


    # Define internal retrievers and their functionality
    def structured_retriever(question: str) -> str:
        entities = entity_chain.invoke({"question": question})
        if not entities.names:
            return "No relevant entities found."

        result = ""
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
                {"query": generate_full_text_query(entity)})
            result += "\n".join([el['output'] for el in response])
        
        return result if result else "No structured data found."


    def retriever(question: str) -> str:
        structured_data = structured_retriever(question)
        unstructured_data = [doc.page_content for doc in vector_index.similarity_search(question)]
        
        if not structured_data and not unstructured_data:
            return "No relevant results found. Please contact customer service."

        final_data = f"""
        **Structured Data:**
        {structured_data or 'No structured data found.'}

        **Unstructured Data:**
        {' '.join(unstructured_data) or 'No unstructured data found.'}
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


    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

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
