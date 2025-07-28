from crewai.tools import BaseTool
from typing import Type, Any, Optional
from pydantic import BaseModel, Field
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import traceback
from src.logger import logging

# Add autopep8 import with fallback
try:
    import autopep8
    def fix_indentation(code: str) -> str:
        try:
            return autopep8.fix_code(code)
        except Exception:
            return code
except ImportError:
    def fix_indentation(code: str) -> str:
        return code

class SeabornPlotInput(BaseModel):
    code: str = Field(..., description="JSON-formatted dictionary string mapping insights to Seaborn code.")
    complete_data: str = Field(..., description="Serialized DataFrame as a JSON string.")

class SeabornPlotGeneratorTool(BaseTool):
    name: str = "Seaborn Plot Generator"
    description: str = (
        "Executes Seaborn plots from a JSON-formatted dictionary string and saves the plots as PNG files "
        "in the 'images' folder. Uses the JSON-formatted dataframe to reconstruct df and "
        "executes plotting code using that DataFrame."
    )
    args_schema: Type[BaseModel] = SeabornPlotInput

    def _run(self, code: str, complete_data: str) -> str:
        try:
            logging.info("=== Starting Seaborn plot generation ===")
            logging.info(f"Input types - code: {type(code)}, complete_data: {type(complete_data)}")
            
            # Convert inputs to strings if they aren't already
            if isinstance(code, dict):
                code = json.dumps(code)
            elif not isinstance(code, str):
                code = str(code)
            
            if isinstance(complete_data, dict):
                complete_data = json.dumps(complete_data)
            elif not isinstance(complete_data, str):
                complete_data = str(complete_data)
            
            logging.info("Inputs converted to strings successfully")
            
            # Parse the DataFrame
            try:
                df = pd.DataFrame(json.loads(complete_data))
                logging.info("DataFrame created successfully.")
            except Exception as e:
                logging.error(f"Error creating DataFrame: {e}")
                return f"Failed to parse complete_data into DataFrame: {e}"

            # Parse the code
            try:
                plots = json.loads(code)
                if not isinstance(plots, dict):
                    raise ValueError("Code must be a JSON string representing a dictionary")
                logging.info(f"Parsed plots keys: {list(plots.keys())}")
            except Exception as e:
                logging.error(f"Error parsing code: {e}")
                return f"Failed to parse code as JSON dictionary: {e}"



            os.makedirs("images", exist_ok=True)
            saved_images = []
            errors = []

            for plot_label, seaborn_code in plots.items():
                if not isinstance(seaborn_code, str):
                    seaborn_code = str(seaborn_code)
                
                # Preserve indentation: only remove trailing whitespace and skip empty lines
                cleaned_code = "\n".join(line.rstrip() for line in seaborn_code.splitlines() if line.strip())
                # Fix indentation before execution
                cleaned_code = fix_indentation(cleaned_code)
                try:
                    logging.info(f"Generating plot for: {plot_label}")
                    plt.clf()
                    plt.close('all')

                    exec_globals = {
                        "df": df,
                        "sns": sns,
                        "plt": plt,
                        "pd": pd,
                        "__builtins__": __builtins__,
                    }

                    exec(cleaned_code, exec_globals)
                    fig = plt.gcf()
                    # Only close the figure, do not save again
                    plt.close(fig)
                    saved_images.append(f"Plot generated for: {plot_label}")
                    logging.info(f"Plot generated for: {plot_label}")

                except Exception as e:
                    error_msg = traceback.format_exc()
                    errors.append(f"Plot '{plot_label}' failed:\n{error_msg}")
                    logging.error(f"Error generating plot '{plot_label}': {error_msg}")

            if saved_images:
                response = f"Plots saved successfully: {saved_images}"
                if errors:
                    response += f"\nHowever, some plots failed:\n" + "\n".join(errors)
                return response
            else:
                return "No plots were saved due to errors:\n" + "\n".join(errors)

        except Exception as e:
            error_msg = traceback.format_exc()
            logging.error(f"Unexpected error in plot generation: {error_msg}")
            return f"Unexpected error in plot generation: {error_msg}"


# class SeabornPlotInput(BaseModel):
#     code: dict = Field(..., description="Dictionary mapping insights to Seaborn code.")
#     complete_data: Any = Field(..., description="DataFrame data as a Python dict (converted from JSON).")

# class SeabornPlotGeneratorTool(BaseTool):
#     name: str = "Seaborn Plot Generator"
#     description: str = (
#         "Executes Seaborn plots from a dictionary mapping insights to Seaborn code and saves the plots as PNG files "
#         "in the 'images' folder. Uses the dictionary to reconstruct the DataFrame."
#     )
#     args_schema: Type[BaseModel] = SeabornPlotInput

#     def _run(self, code: dict, complete_data: Any) -> str:
#         try:
#             logging.info("Starting Seaborn plot generation...")
#             logging.info(f"Received code keys: {list(code.keys())}")
#             df = pd.DataFrame(complete_data)
#             logging.info("DataFrame created successfully.")
#         except Exception as e:
#             return f"Failed to parse complete_data into DataFrame: {e}"

#         os.makedirs("images", exist_ok=True)
#         saved_images = []
#         errors = []

#         for plot_label, seaborn_code in code.items():
#             cleaned_code = "\n".join(line.strip() for line in seaborn_code.splitlines() if line.strip())
#             try:
#                 logging.info(f"Generating plot for: {plot_label}")
#                 plt.clf()
#                 plt.close('all')

#                 exec_globals = {
#                     "df": df,
#                     "sns": sns,
#                     "plt": plt,
#                     "__builtins__": __builtins__,
#                 }

#                 exec(str(cleaned_code), exec_globals)
#                 fig = plt.gcf()

#                 file_name = f"{plot_label[:50].replace(' ', '_').replace('/', '_')}.png"
#                 file_path = os.path.join("images", file_name)
#                 fig.savefig(file_path, format='png', bbox_inches='tight')
#                 plt.close(fig)
#                 saved_images.append(file_path)
#                 logging.info(f"Plot saved successfully: {file_path}")

#             except Exception as e:
#                 error_msg = traceback.format_exc()
#                 errors.append(f"Plot '{plot_label}' failed:\n{error_msg}")

#         if saved_images:
#             response = f"Plots saved successfully: {saved_images}"
#             if errors:
#                 response += f"\nHowever, some plots failed:\n" + "\n".join(errors)
#             return response
#         else:
#             return "No plots were saved due to errors:\n" + "\n".join(errors)



