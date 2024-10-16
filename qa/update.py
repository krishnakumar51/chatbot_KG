import os
import logging
import warnings
from typing import List, Tuple
from dotenv import load_dotenv
from neo4j import GraphDatabase, exceptions
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
from langchain.chains import  create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import re

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
        # llm = ChatGroq(groq_api_key=api_key, model_name="llama3-groq-70b-8192-tool-use-preview")
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


class Entities(BaseModel):
    names: List[str] = Field(
        ...,
        description="All person, organization, or business entities that appear in the text."
    )


from langchain.prompts import ChatPromptTemplate

def create_chain(llm, graph, vector_index):
    """
    Create and return a QA chain using LLM, Neo4j graph, and vector index. 
    This chain handles chat history, document retrieval, and question answering.
    """

    # Step 1: Entity extraction prompt setup using structured output from LLM
    entity_prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You are a concise and polite assistant, designed to strictly extract important entity names such as people, organizations, and businesses from text. "
        "Your response must only include the names without any extra information, context, or links. "
        "Return the information in a structured JSON format. "
        "Example format: {\"names\": [\"Entity1\", \"Entity2\"]}. "
        "Do not provide answers from your own knowledge. Only respond with entity names that are directly retrieved from the given text or knowledge graph."
        ),
        ("human", "{input}")
    ])
    
    # Step 2: Create entity extraction chain using LLM with structured output mode
    entity_chain = entity_prompt | llm.with_structured_output(Entities, method="json_mode")

    # Step 3: Set up the system prompt template to accept both context and input
    system_prompt_template = ChatPromptTemplate.from_template(
        template="""
        You are an assistant for question-answering tasks. Use the provided context and chat history to answer the question directly. 
        If you don't know the answer, respond with 'I don't know.' Keep the response concise.
        """,
        input_variables=["context", "input"]  # Specify the required input variables here
    )

    # Step 4: Create the question-answering chain using the updated prompt template
    question_answer_chain = create_stuff_documents_chain(llm, system_prompt_template)

    # Step 5: Set up the retriever to fetch relevant documents from the Neo4j vector index
    retriever = vector_index.as_retriever()

    # Step 6: Create the full retrieval-augmented generation (RAG) chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Step 7: Create function to handle chat history and query once for RAG chain
    def _get_condensed_query_with_history(chat_history, question):
        """
        Combines chat history and current question into a final input for the RAG chain.
        """
        condensed_question = condense_prompt.invoke({
            "chat_history": chat_history,
            "question": question
        })["question"]

        response = rag_chain.invoke({
            "input": condensed_question  # The rephrased question
        })
        return response["answer"]

    # Step 8: Structured retriever for querying the Neo4j graph using entities
    def structured_retriever(question: str) -> str:
        """
        Retrieves structured data from the Neo4j graph based on the entities extracted from the question.
        """
        entities = entity_chain.invoke({"input": question})
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
                MATCH (node)-[r:MENTIONS]->(neighbor)
                RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                UNION ALL
                WITH node
                MATCH (node)<-[r:MENTIONS]-(neighbor)
                RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 50
                """, 
                {"query": entity}
            )
            result += "\n".join([el['output'] for el in response])
        
        return result if result else "No structured data found."

    # Step 9: General retriever combining structured and unstructured data
    def retriever(question: str) -> str:
        """
        Combines both structured and unstructured data retrieval to answer the user's question.
        """
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

    # Final return: The entire chain structure with history-awareness, entity extraction, and retrieval
    return {
        "chain": rag_chain,                      # The history-aware RAG chain
        "condense_query": _get_condensed_query_with_history,  # Condensed follow-up questions for standalone query
        "structured_retriever": structured_retriever, # Neo4j graph data retriever using entities
        "retriever": retriever                  # Combines structured and unstructured data
    }





def remove_problematic_chars(text: str) -> str:
    """
    Remove problematic characters that may interfere with Lucene-based queries.
    Specifically removing slashes and other special characters that aren't necessary.
    """
    # Remove forward slashes
    text = text.replace("/", "")
    
    # Optionally handle backslashes and other special characters if needed
    text = text.replace("\\", "")
    
    # Remove other problematic special characters like *, %, etc. if necessary
    text = re.sub(r'[!@#$%^&*()_+={}\[\]:;"\'<>?,~`]', '', text)

    # Optionally handle additional sanitization logic (e.g., extra spaces, etc.)
    text = ' '.join(text.split())
    
    return text

def sanitize_input(question: str) -> str:
    """
    Sanitize user input to handle problematic characters by removing slashes,
    quotes, and other special symbols that can cause issues with Lucene queries.
    """
    # Remove problematic characters
    sanitized_question = remove_problematic_chars(question)
    
    return sanitized_question


def format_urls_as_markdown(text):
    """
    Convert all URLs in the text to Markdown format [Click here](url).
    """
    url_pattern = r'(http[s]?://[^\s]+|www\.[^\s]+)'

    def replace_with_markdown(match):
        url = match.group(0)
        return f"[Click here]({url})"

    formatted_text = re.sub(url_pattern, replace_with_markdown, text)
    return formatted_text


def get_qa_response(chain, question: str, chat_history: List[dict]) -> str:
    """
    Get an answer for the user's query using the pre-defined RAG chain.
    This includes chat history, entity extraction, and document retrieval.
    """
    # Sanitize the input question to avoid issues with special characters
    sanitized_question = sanitize_input(question)
    
    print(f"Sanitized input question: {sanitized_question}")
    
    try:
        # Invoke the RAG chain with the sanitized question and chat history
        response = chain["chain"].invoke({
            "input": sanitized_question,          # The sanitized question
            "chat_history": chat_history          # Chat history to provide context
        })
        
        # Check if there are any results from the chain
        if not response or "answer" not in response:
            return (
                "**No relevant results found.**\n\n"
                "Please reach out to our customer service."
            )
        
        # Format the final result (including handling any URLs in the response)
        formatted_result = format_urls_as_markdown(response["answer"])
        
        return f"**Response:**\n\n{formatted_result}"

    except ServiceUnavailable:
        return (
            "**Sorry, the database is currently unavailable.**\n\n"
            "Please try again later. For immediate assistance, reach out to customer support."
        )

    except Exception as e:
        return (
            f"**An error occurred:** `{str(e)}`\n\n"
            "For assistance, please contact our customer service."
        )


