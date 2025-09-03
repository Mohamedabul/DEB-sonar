
import streamlit as st
import os
import time
from src.effort_tracker import EffortTracking
from src.utils import clear_analysis_outputs
import shutil
from src.logger import logging
 
st.set_page_config(layout="wide")
 
# Custom CSS
st.markdown("""
    <style>
    .main .block-container {
        max-width: 95vw !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    .element-container:has(.stChatMessage) {
        width: 100% !important;
    }
    .stChatMessage {
        width: 100% !important;
        max-width: 100% !important;
    }
    .stChatMessage .stMarkdown, .stChatMessage .stCodeBlock {
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }
    .stCodeBlock > pre {
        overflow-x: auto;
    }
    </style>
""", unsafe_allow_html=True)
 
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
 
# Session defaults
default_session_values = {
    'uploaded_file': None,
    'selected_project': None,
    'result': None,
    'show_chat': False,
    'messages': []
}
 
for key, default in default_session_values.items():
    if key not in st.session_state:
        st.session_state[key] = default
 
# File Upload UI
with st.expander("Upload File"):
    uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])
    if uploaded_file is not None:
        with st.spinner("Uploading..."):
            time.sleep(1)
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_file = file_path
            st.success("Upload complete!")
 
# App Title and Project Selection
st.title("Welcome to DEB Pilot")
 
selected_project = st.selectbox(
    "Select Project",
    ["", "Orchestration", "Adoption", "Observability","Backstage","CEP Portal"],
    index=0,
    key="selected_project"
)
 
if selected_project:
    st.session_state.show_chat = True
 
if st.session_state.show_chat:
    st.subheader("Chat with Effort Analyzer")
 
    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.code(msg["content"], language="text")
 
    # Chat input
    user_input = st.chat_input("Ask a question about the Effort Analysis Report...")
    if user_input:
        clear_analysis_outputs()
        st.session_state.messages.append({"role": "user", "content": user_input})
 
        with st.spinner("Analyzing..."):
            effort_tracker = EffortTracking(selected_project.lower())
            response = effort_tracker.effort_analysis(user_input)
 
        st.session_state.messages.append({"role": "assistant", "content": response})
 
        with st.chat_message("user"):
            st.code(user_input, language="text")
        with st.chat_message("assistant"):
            if response.strip() == "Analysis Complete":
                report_path = os.path.abspath("final_analysis_report.md")
                logging.info(f"Looking for report at: {report_path}")
 
                if os.path.exists(report_path):
                    try:
                        with open(report_path, "r", encoding="utf-8") as f:
                            md_content = f.read()
                        logging.info("Final Analysis Report Found")
 
                        # Render the report line by line
                        st.markdown("### Final Effort Analysis Report", unsafe_allow_html=False)
 
                        lines = md_content.splitlines()
                        buffer = []
                        i = 0
                        while i < len(lines):
                            line = lines[i].rstrip()
 
                            # Handle images
                            if line.startswith("![](") and line.endswith(")"):
                                if buffer:
                                    st.markdown("\n".join(buffer), unsafe_allow_html=True)
                                    buffer = []
                                image_path = line[4:-1]
                                if os.path.exists(image_path):
                                    st.image(image_path, width=1000)
                                else:
                                    st.warning(f"Image not found: {image_path}")
                                i += 1
                                continue
 
                            # Detect table: line with pipes AND next line is table header separator
                            is_table_start = (
                                "|" in line and
                                i + 1 < len(lines)
                            )
 
                            if is_table_start:
                                if buffer:
                                    st.markdown("\n".join(buffer), unsafe_allow_html=True)
                                    buffer = []
 
                                # Start capturing table
                                table_lines = [line]
                                j = i + 1
                                while j < len(lines) and "|" in lines[j]:
                                    table_lines.append(lines[j])
                                    j += 1
                                st.code("\n".join(table_lines), language="text")
                                i = j  # Advance i to after the table
                                continue
 
                            # Regular paragraph lines
                            if line.strip():
                                buffer.append(line)
                            else:
                                if buffer:
                                    st.markdown("\n".join(buffer), unsafe_allow_html=True)
                                    buffer = []
                            i += 1
 
                        # Flush any remaining content
                        if buffer:
                            st.markdown("\n".join(buffer), unsafe_allow_html=True)
 
                    except Exception as e:
                        logging.error(f"Error reading report: {e}")
                        st.error("Could not read the report file.")
                else:
                    st.info("No final analysis report found.")
            else:
                st.code(response, language="text")
