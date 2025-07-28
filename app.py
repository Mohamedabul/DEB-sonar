# import streamlit as st
# import os
# import time
# from src.effort_tracker import EffortTracking
# from src.utils import clear_analysis_outputs, convert_md_to_pdf, get_download_link
# import shutil
# from src.logger import logging
# from weasyprint import HTML, CSS
# import markdown
# import base64
# from PIL import Image
# import io

# # Set font configuration for WeasyPrint
# if os.name == 'nt':  # Windows
#     os.environ['FONTCONFIG_PATH'] = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'fontconfig')
#     os.environ['FONTCONFIG_FILE'] = os.path.join(os.environ['FONTCONFIG_PATH'], 'fonts.conf')

# st.set_page_config(layout="wide")

# # Custom CSS
# st.markdown("""
#     <style>
#     .main .block-container {
#         max-width: 95vw !important;
#         padding-left: 2rem !important;
#         padding-right: 2rem !important;
#     }
#     .element-container:has(.stChatMessage) {
#         width: 100% !important;
#     }
#     .stChatMessage {
#         width: 100% !important;
#         max-width: 100% !important;
#     }
#     .stChatMessage .stMarkdown, .stChatMessage .stCodeBlock {
#         white-space: pre-wrap !important;
#         word-break: break-word !important;
#     }
#     .stCodeBlock > pre {
#         overflow-x: auto;
#     }
#     </style>
# """, unsafe_allow_html=True)

# DATA_DIR = "data"
# os.makedirs(DATA_DIR, exist_ok=True)

# # Session defaults
# default_session_values = {
#     'uploaded_file': None,
#     'selected_project': None,
#     'result': None,
#     'show_chat': False,
#     'messages': []
# }

# for key, default in default_session_values.items():
#     if key not in st.session_state:
#         st.session_state[key] = default

# # File Upload UI
# with st.expander("Upload File"):
#     uploaded_file = st.file_uploader("Choose a file", type=["xlsx"])
#     if uploaded_file is not None:
#         with st.spinner("Uploading..."):
#             time.sleep(1)
#             file_path = os.path.join(DATA_DIR, uploaded_file.name)
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
#             st.session_state.uploaded_file = file_path
#             st.success("Upload complete!")

# # App Title and Project Selection
# st.title("Welcome to DEB Pilot")

# selected_project = st.selectbox(
#     "Select Project",
#     ["", "Orchestration", "Adoption", "Observability","Backstage","CEP Portal"],
#     index=0,
#     key="selected_project"
# )

# if selected_project:
#     st.session_state.show_chat = True

# if st.session_state.show_chat:
#     st.subheader("Chat with Effort Analyzer")

#     # Display message history
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.code(msg["content"], language="text")

#     # Chat input
#     user_input = st.chat_input("Ask a question about the Effort Analysis Report...")
#     if user_input:
#         clear_analysis_outputs()
#         st.session_state.messages.append({"role": "user", "content": user_input})

#         with st.spinner("Analyzing..."):
#             effort_tracker = EffortTracking(selected_project.lower())
#             response = effort_tracker.effort_analysis(user_input)

#         st.session_state.messages.append({"role": "assistant", "content": response})

#         with st.chat_message("user"):
#             st.code(user_input, language="text")
#         with st.chat_message("assistant"):
#             if response.strip() == "Analysis Complete":
#                 report_path = os.path.abspath("final_analysis_report.md")
#                 logging.info(f"Looking for report at: {report_path}")

#                 if os.path.exists(report_path):
#                     try:
#                         with open(report_path, "r", encoding="utf-8") as f:
#                             md_content = f.read()
#                         logging.info("Final Analysis Report Found")

#                         # Render the report line by line
#                         st.markdown("### Final Effort Analysis Report", unsafe_allow_html=False)

#                         lines = md_content.splitlines()
#                         buffer = []
#                         i = 0
#                         while i < len(lines):
#                             line = lines[i].rstrip()

#                             # Handle images
#                             if line.startswith("![](") and line.endswith(")"):
#                                 if buffer:
#                                     st.markdown("\n".join(buffer), unsafe_allow_html=True)
#                                     buffer = []
#                                 image_path = line[4:-1]
#                                 if os.path.exists(image_path):
#                                     st.image(image_path, width=1000)
#                                 else:
#                                     st.warning(f"Image not found: {image_path}")
#                                 i += 1
#                                 continue

#                             # Detect table: line with pipes AND next line is table header separator
#                             is_table_start = (
#                                 "|" in line and
#                                 i + 1 < len(lines)
#                             )

#                             if is_table_start:
#                                 if buffer:
#                                     st.markdown("\n".join(buffer), unsafe_allow_html=True)
#                                     buffer = []

#                                 # Start capturing table
#                                 table_lines = [line]
#                                 j = i + 1
#                                 while j < len(lines) and "|" in lines[j]:
#                                     table_lines.append(lines[j])
#                                     j += 1
#                                 st.code("\n".join(table_lines), language="text")
#                                 i = j  # Advance i to after the table
#                                 continue

#                             # Regular paragraph lines
#                             if line.strip():
#                                 buffer.append(line)
#                             else:
#                                 if buffer:
#                                     st.markdown("\n".join(buffer), unsafe_allow_html=True)
#                                     buffer = []
#                             i += 1

#                         # Flush any remaining content
#                         if buffer:
#                             st.markdown("\n".join(buffer), unsafe_allow_html=True)

#                         # Add PDF download button
#                         if os.path.exists(report_path):
#                             try:
#                                 # Get all image paths from the markdown content
#                                 image_paths = []
#                                 for line in lines:
#                                     if line.startswith("![](") and line.endswith(")"):
#                                         image_path = line[4:-1]
#                                         if os.path.exists(image_path):
#                                             image_paths.append(image_path)

#                                 # Convert markdown to PDF
#                                 pdf_bytes = convert_md_to_pdf(md_content, image_paths)
                                
#                                 # Create download link
#                                 st.markdown(
#                                     get_download_link(pdf_bytes, "effort_analysis_report.pdf"),
#                                     unsafe_allow_html=True
#                                 )
#                             except Exception as e:
#                                 logging.error(f"Error generating PDF: {e}")
#                                 st.error("Could not generate PDF report.")

#                     except Exception as e:
#                         logging.error(f"Error reading report: {e}")
#                         st.error("Could not read the report file.")
#                 else:
#                     st.info("No final analysis report found.")
#             else:
#                 st.code(response, language="text")

# # def convert_md_to_pdf(md_content, image_paths):
# #     # Convert markdown to HTML
# #     html_content = markdown.markdown(md_content, extensions=['tables'])
    
# #     # Add custom CSS for better PDF formatting
# #     css = CSS(string='''
# #         body { font-family: Arial, sans-serif; }
# #         table { border-collapse: collapse; width: 100%; margin: 20px 0; }
# #         th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
# #         th { background-color: #f2f2f2; }
# #         img { max-width: 100%; height: auto; }
# #     ''')
    
# #     # Create HTML document with images
# #     html = f"""
# #     <html>
# #         <head>
# #             <meta charset="UTF-8">
# #             <title>Effort Analysis Report</title>
# #         </head>
# #         <body>
# #             <h1>Effort Analysis Report</h1>
# #             {html_content}
# #         </body>
# #     </html>
# #     """
    
# #     # Generate PDF
# #     pdf = HTML(string=html).write_pdf(stylesheets=[css])
# #     return pdf

# # def get_download_link(pdf_bytes, filename):
# #     b64 = base64.b64encode(pdf_bytes).decode()
# #     href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'
# #     return href

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
