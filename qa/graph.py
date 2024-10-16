from neo4j import GraphDatabase
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
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from typing import List
import logging
import warnings
from neo4j.exceptions import ServiceUnavailable
from langchain_groq import ChatGroq
import streamlit as st 
import toml



# Logging configuration
logging.getLogger("neo4j").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables
load_dotenv()



try:
    # Local development using secrets.toml or environment variables
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    api_key = os.getenv("GROQ_API_KEY")
    print("env")
except KeyError:
    # Deployment on Streamlit Cloud
    api_key = st.secrets["general"]["GROQ_API_KEY"]
    uri = st.secrets["general"]["NEO4J_URI"]
    username = st.secrets["general"]["NEO4J_USERNAME"]
    password = st.secrets["general"]["NEO4J_PASSWORD"]
    print("secerts")



llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192")

# Neo4j graph setup
graph = Neo4jGraph(url=uri, username=username, password=password)

# Embeddings setup for hybrid search
embeddings_hf = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_index = Neo4jVector.from_existing_graph(
    embedding=embeddings_hf,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# Full-text query generation function
def generate_full_text_query(input: str) -> str:
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Entity extraction model using Pydantic and LLM
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

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

# Function to retrieve structured data from Neo4j
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

# Function to retrieve structured and unstructured data
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
        | ChatGroq(model="llama3-8b-8192",temperature=0,)
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

# Function to handle the QA process
def get_qa_response(question: str) -> str:
    try:
        result = chain.invoke({"question": question})
        
        if not result:
            # Return Markdown formatted text with a WhatsApp support link
            return (
                "**No relevant results found.**\n\n"
                "Please reach out to our customer service on WhatsApp: "
                "[Contact Support](https://api.whatsapp.com/send/?phone=9105575000)"
            )
        
        # If result is found, return it wrapped in a Markdown-friendly format
        return f"**Response:**\n\n{result}"
    
    except ServiceUnavailable:
        # If the database is unavailable, redirect to customer support
        return (
            "**Sorry, the database is currently unavailable.**\n\n"
            "Please try again later. For immediate assistance, please reach out to our customer service on WhatsApp: "
            "[Contact Support](https://api.whatsapp.com/send/?phone=9105575000)"
        )
    
    except Exception as e:
        
        # Return Markdown formatted error message
        return (
            f"**An error occurred:** "
            "For assistance, please reach out to our customer service on WhatsApp: "
            "[Contact Support](https://api.whatsapp.com/send/?phone=9105575000)"
        )
