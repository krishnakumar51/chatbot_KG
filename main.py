import streamlit as st
from qa.ft import get_qa_response, initialize_system, create_chain
import os

# Set up Streamlit page configurations
st.set_page_config(page_title="QA Chatbot", page_icon="🤖")
st.title("🤖 QA Chatbot")

# Initialize models and graph only if they haven't been loaded before
if "initialized" not in st.session_state:
    st.session_state.llm, st.session_state.graph, st.session_state.vector_index = initialize_system()
    st.session_state.chain = create_chain(st.session_state.llm, st.session_state.graph, st.session_state.vector_index)
    st.session_state.initialized = True
    print("Environment variables loaded and components initialized.")

# Initialize the session state for conversation history and status flags
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "user_input_processed" not in st.session_state:
    st.session_state.user_input_processed = False

if "api_called" not in st.session_state:
    st.session_state.api_called = False

# Display the existing conversation messages
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box for the user to provide queries or prompts
if prompt := st.chat_input("How can I assist you today?"):
    if not st.session_state.user_input_processed:
        # Append the user's message to the conversation history
        st.session_state.conversation_history.append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)

        # Set flags indicating that the user input has been processed
        st.session_state.user_input_processed = True
        st.session_state.api_called = False  # Reset API call status for this input

# Generate a response only if input is provided and the API hasn't been called yet
if st.session_state.user_input_processed and not st.session_state.api_called:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get the last user query and generate a response using the QA system
                last_user_message = st.session_state.conversation_history[-1]["content"]
                
                # Call `get_qa_response` with the required arguments: `vector_index`, `chain`, and `question`
                response =get_qa_response(st.session_state.chain, last_user_message)

                # Display the assistant's response and update the conversation history
                st.markdown(response)
                st.session_state.conversation_history.append({"role": "assistant", "content": response})

                # Mark API call as completed
                st.session_state.api_called = True
                st.session_state.user_input_processed = False  # Reset flag for the next user input

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Sidebar controls to clear the chat history
st.sidebar.title("🛠️ Options")
if st.sidebar.button("Clear Conversation History"):
    st.session_state.conversation_history = []
    st.sidebar.success("Conversation history cleared successfully!")

# Sidebar information or app description
st.sidebar.markdown(
    """
    ### About the Chatbot
    This chatbot is designed to interact with users using a custom QA system.
    You can ask it questions, and it will provide responses based on the data it has been trained on.
    """
)
