# import os
# import time
# import streamlit as st
# from src.effort_tracker import EffortTracking

# # Create a folder for storing uploaded files
# DATA_DIR = "data"
# os.makedirs(DATA_DIR, exist_ok=True)

# # Session State Initialization
# if 'uploaded_file' not in st.session_state:
#     st.session_state.uploaded_file = None
# if 'selected_project' not in st.session_state:
#     st.session_state.selected_project = None
# if 'result' not in st.session_state:
#     st.session_state.result = None
# if 'show_chat' not in st.session_state:
#     st.session_state.show_chat = False
# if 'messages' not in st.session_state:
#     st.session_state.messages = []  # Stores chat messages

# # Upload File Section
# with st.expander("Upload File"):
#     uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])
#     if uploaded_file is not None:
#         with st.spinner("Uploading..."):
#             time.sleep(1)  # Simulate upload delay
#             file_path = os.path.join(DATA_DIR, uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
#             st.session_state.uploaded_file = file_path
#             st.success("Upload complete!")

# # Main Application UI
# st.title("Welcome to DEB Pilot")

# selected_project = st.selectbox(
#     "Select Project", ["", "Orchestration", "Adoption", "Observability"],
#     index=0, key="selected_project"
# )

# if selected_project:
#     with st.spinner("Generating Effort Analysis Report..."):
#         effort_tracker = EffortTracking(selected_project.lower())
#         st.session_state.result = effort_tracker.burn_down_generation()

# if st.session_state.result is not None:
#     st.write(f"Effort Analysis Report for {selected_project}:")
#     st.write(st.session_state.result)

#     if st.button('Perform Analysis'):
#         st.session_state.show_chat = True  # Show chat UI

# # Chat Interface
# if st.session_state.show_chat:
#     st.subheader("Chat with Effort Analyzer")

#     # Display chat messages from history
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"])

#     # User Input
#     user_input = st.chat_input("Ask a question about the Effort Analysis Report...")
#     if user_input:
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": user_input})

#         # Perform LLM analysis on user query
#         with st.spinner("Analyzing..."):
#             # response = effort_tracker.effort_analysis(user_input, st.session_state.result)
#             response = effort_tracker.effort_analysis(user_input)

#         # Add assistant response to chat history
#         st.session_state.messages.append({"role": "assistant", "content": response})

#         # Display the latest interaction
#         with st.chat_message("user"):
#             st.markdown(user_input)
#         with st.chat_message("assistant"):
#             st.markdown(response)






import streamlit as st
import os
import time
from src.effort_tracker import EffortTracking

# Create a folder for storing uploaded files
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Session State Initialization
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False
if 'messages' not in st.session_state:
    st.session_state.messages = []  # Stores chat messages

# Upload File Section
with st.expander("Upload File"):
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])
    if uploaded_file is not None:
        with st.spinner("Uploading..."):
            time.sleep(1)  # Simulate upload delay
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_file = file_path
            st.success("Upload complete!")

# Main Application UI
st.title("Welcome to DEB Pilot")

selected_project = st.selectbox(
    "Select Project", ["", "Orchestration", "Adoption", "Observability"],
    index=0, key="selected_project"
)

# if selected_project:
#     with st.spinner("Generating Effort Analysis Report..."):
#         effort_tracker = EffortTracking(selected_project.lower())
#         st.session_state.result = effort_tracker.burn_down_generation()


if selected_project:
    st.session_state.show_chat = True

# Chat Interface
if st.session_state.show_chat:
    st.subheader("Chat with Effort Analyzer")

    # Display chat messages from history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.code(msg["content"], language="text")

    # User Input
    user_input = st.chat_input("Ask a question about the Effort Analysis Report...")
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Perform LLM analysis on user query
        with st.spinner("Analyzing..."):
            # response = effort_tracker.effort_analysis(user_input, st.session_state.result)
            effort_tracker = EffortTracking(selected_project.lower())
            report = effort_tracker.effort_analysis(user_input)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": report})

        # Display the latest interaction
        with st.chat_message("user"):
            st.code(user_input, language="text")
        with st.chat_message("assistant"):
            st.code(report, language="text")


