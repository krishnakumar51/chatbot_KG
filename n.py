import streamlit as st
from qa.history import get_qa_response, initialize_system, create_chain
import os
from datetime import datetime

# Set up Streamlit page configurations
st.set_page_config(page_title="QA Chatbot", page_icon="🤖")
st.title("🤖 QA Chatbot")

# Function to create or append to the conversation file
def save_conversation_to_file():
    folder = "conversation_logs"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create or open the file for appending messages
    session_filename = st.session_state['session_file']

    with open(session_filename, "a", encoding="utf-8") as file:
        for message in st.session_state.conversation_history[len(st.session_state.file_written_history):]:
            role = "You" if message["role"] == "human" else "Assistant"
            file.write(f"{role}: {message['content']}\n")
        
        # Update the list of messages that have been written to the file
        st.session_state.file_written_history = list(st.session_state.conversation_history)

# Ask for user name and phone number if not already provided
if "user_name" not in st.session_state or "user_phone" not in st.session_state:
    with st.form("user_info_form"):
        user_name = st.text_input("Enter your name")
        user_phone = st.text_input("Enter your phone number")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if user_name and user_phone:
                st.session_state.user_name = user_name
                st.session_state.user_phone = user_phone
                
                # Create a unique filename for this session
                session_filename = f"conversation_logs/{st.session_state.user_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
                st.session_state['session_file'] = session_filename

                # Write user info at the start of the file
                with open(session_filename, "w", encoding="utf-8") as file:
                    file.write(f"User Name: {st.session_state.user_name}\n")
                    file.write(f"Phone Number: {st.session_state.user_phone}\n\n")

                # Initialize history tracking for written messages
                st.session_state['file_written_history'] = []

                st.success(f"Welcome {st.session_state.user_name}!")
            else:
                st.error("Please provide both name and phone number.")
                st.stop()  # Stop further execution until user input is valid
else:
    st.success(f"Welcome back, {st.session_state.user_name}!")

# Initialize models and graph only if they haven't been loaded before
if "initialized" not in st.session_state:
    st.session_state.llm, st.session_state.graph, st.session_state.vector_index, st.session_state.chat = initialize_system()
    st.session_state.chain = create_chain(st.session_state.llm, st.session_state.graph, st.session_state.vector_index, st.session_state.chat)
    st.session_state.initialized = True
    print("Environment variables loaded and components initialized.")

# Initialize the session state for conversation history and status flags
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []  # Chat history

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

                # Ensure the input isn't empty or invalid
                if last_user_message.strip():
                    # Call `get_qa_response` with the chain, question, and chat history
                    response = get_qa_response(
                        st.session_state.chain,              # Chain with entity and history awareness
                        last_user_message,                   # User's latest question
                        st.session_state.conversation_history  # Chat history to provide context
                    )

                    # Display the assistant's response and update the conversation history
                    st.markdown(response)
                    st.session_state.conversation_history.append({"role": "assistant", "content": response})

                    # Mark API call as completed
                    st.session_state.api_called = True
                    st.session_state.user_input_processed = False  # Reset flag for the next user input

                    # Save the conversation to file
                    save_conversation_to_file()
                else:
                    st.error("Please enter a valid question.")

            except Exception as e:
                st.error("An error occurred. Please try again later.")
                # Optionally, log the error for debugging purposes
                print(f"Error: {str(e)}")

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
