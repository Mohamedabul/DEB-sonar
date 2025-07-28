import warnings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
from src.logger import logging
from src.exception_handler import CustomException
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.prompts import PromptTemplate
import ast
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import sys
from src.llm_initialize import LLMInitializer
from src.data_fetching import DataFetching
from dataclasses import dataclass
from src.prompts import EffortAnalysisPrompt,EffortAnalysisPrompt2, Router
from src.tools.graph_plotting import SeabornPlotGeneratorTool, SeabornPlotInput
from langchain_core.output_parsers import StrOutputParser
import numpy as np
from crewai import Agent, Task, Crew, Process
from crewai_tools import FileReadTool,DirectoryReadTool
from IPython.display import Markdown
import json
from textwrap import dedent
import re
import time

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="crewai_tools")
warnings.filterwarnings("ignore", message="Fontconfig error: Cannot load default config file")

class PrintThoughtLogger:
    def __init__(self):
        self.steps = []

    def __call__(self, step_output):
        # Handle AgentFinish object
        if hasattr(step_output, 'return_values'):
            output = step_output.return_values.get('output', '')
            log = step_output.return_values.get('log', '')
            print("\n--- Agent Step ---")
            if log:
                print(f"🧠 Thought Process: {log}")
            print(f"✅ Final Output: {output}")
            print("------------------")
            return

        # Handle dictionary output
        if isinstance(step_output, dict):
            thought = step_output.get("thought")
            action = step_output.get("action")
            observation = step_output.get("observation")
            output = step_output.get("output")
            log = step_output.get("log")

            # Store if needed
            self.steps.append(step_output)

            # Display to user (or write to file/log)
            print("\n--- Agent Step ---")
            if log:
                print(f"🧠 Thought Process: {log}")
            if thought:
                print(f"🧠 Thought: {thought}")
            if action:
                print(f"🔧 Action: {action}")
            if observation:
                print(f"📦 Observation: {observation}")
            if output:
                print(f"✅ Final Output: {output}")
            print("------------------")
        else:
            # Handle other types of output
            print("\n--- Agent Step ---")
            print(f"Output: {str(step_output)}")
            print("------------------")

@dataclass
class EffortTrackingConfig:   
    def __init__(self, file_name):
        self.llm_init = LLMInitializer()
        # self.llm1 = self.llm_init.initialize_llm(model_num=1)
        # self.llm2 = self.llm_init.initialize_llm(model_num=2, temperature=0.0)
        self.llm3 = self.llm_init.initialize_llm(model_num=3, temperature=0.0)
        self.llm4 = self.llm_init.initialize_llm(model_num=4, temperature=0.3)
        self.llm5 = self.llm_init.initialize_llm(model_num=5, temperature=0.3)
        self.data_fetching = DataFetching(file_name)
        self.df =  self.data_fetching.data_fetching()

    

class EffortTracking:
    def __init__(self, file_name):
        self.file_name = file_name
        self.config = EffortTrackingConfig(file_name)
    
    def effort_analysis(self,question):
        try:
            print('Effort Analysis Started',datetime.now())
            start = datetime.now()
            logging.info('Effort Analysis Started')

            df = self.config.df
            complete_data = df.to_json(orient='records')
            # print(df.columns)
             
            df2 = df[['Project / Workstream Name', 'Deliverable', 'Variance', 'EAC (Estimate at Completion)', 'ETC (Estimate to Complete)', 'Actual Effort Hours', 'Actual End Date', 'Actual Start Date', 'Estimated Effort (Hours)', 'Sprint', 'Estimated\n Start Date', 'Estimated\n End Date', 'Capacity', 'Team Member']]

            # Preprocess Sprint column to extract sprint numbers
            df2['Sprint'] = df2['Sprint'].str.extract('(\d+)').astype(int)

            df3 = pd.DataFrame()
            df3['Sum of Capacity'] = df2.groupby(['Sprint','Team Member'])['Capacity'].sum()
            df3['Sum of Estimated Effort'] = df2.groupby(['Sprint','Team Member'])['Estimated Effort (Hours)'].sum()
            df3['Sum of Actual Effort'] = df2.groupby(['Sprint','Team Member'])['Actual Effort Hours'].sum()
            df3['Sum of ETC (Estimate to Complete)'] = df2.groupby(['Sprint','Team Member'])['ETC (Estimate to Complete)'].sum()
            df3 = df3.reset_index()

            df4 = pd.DataFrame()
            df4['Sum of Capacity'] = df2.groupby('Sprint')['Capacity'].sum()
            df4['Sum of Estimated Effort'] = df2.groupby('Sprint')['Estimated Effort (Hours)'].sum()
            df4['Sum of Actual Effort'] = df2.groupby('Sprint')['Actual Effort Hours'].sum()
            df4['Sum of ETC (Estimate to Complete)'] = df2.groupby('Sprint')['ETC (Estimate to Complete)'].sum()
            df4 = df4.reset_index()
            df4['Team Member'] = None

            df_concat = pd.concat([df3, df4], ignore_index=True)

            df_sorted = df_concat.sort_values(by=['Sprint', 'Team Member'], ascending=[True,True])

            grand_total = df4.drop(columns=['Sprint', 'Team Member']).sum()

            grand_total_row = pd.DataFrame({
                'Sprint': ['Grand Total'], 
                'Team Member': [None], 
                **grand_total.to_dict()
            })

            df_final = pd.concat([df_sorted, grand_total_row], ignore_index=True)
            # print(df_final)

            df2_copy = df2.copy()
            df2_copy['Estimated\n Start Date'] = pd.to_datetime(df2_copy['Estimated\n Start Date'], errors='coerce')
            df2_copy['Estimated\n End Date'] = pd.to_datetime(df2_copy['Estimated\n End Date'], errors='coerce')
            
            current_date = pd.Timestamp.now()
            
            df_status = pd.DataFrame()
            # Custom aggregation: if any NaT in group, result is NaT; else min/max
            def safe_min(series):
                return pd.NaT if series.isna().any() else series.min()
            def safe_max(series):
                return pd.NaT if series.isna().any() else series.max()
            df_status['Sprint_Start'] = df2_copy.groupby('Sprint')['Estimated\n Start Date'].agg(safe_min)
            df_status['Sprint_End'] = df2_copy.groupby('Sprint')['Estimated\n End Date'].agg(safe_max)
            
            def calculate_status(row):
                start_date = row['Sprint_Start']
                end_date = row['Sprint_End']
                
                if pd.isna(start_date):
                    return 'Not Started'
                if not isinstance(start_date, pd.Timestamp):
                    start_date = pd.to_datetime(start_date)
                if not isinstance(end_date, pd.Timestamp) and not pd.isna(end_date):
                    end_date = pd.to_datetime(end_date)
                if start_date <= current_date and pd.isna(end_date):
                    return 'In Progress'
                if pd.isna(end_date):
                    return 'Not Started'
                if start_date > current_date and end_date > current_date:
                    return 'Not Started'
                elif start_date <= current_date and end_date > current_date:
                    return 'In Progress'
                elif start_date <= current_date and end_date <= current_date:
                    return 'Completed'
                else:
                    return 'Not Started'
            
            df_status['Status'] = df_status.apply(calculate_status, axis=1)
            df_status = df_status.reset_index()
            # print(df_status)
            
            df_final = df_final.merge(
                df_status[['Sprint', 'Status']], 
                on='Sprint', 
                how='left'
            )
            
            df_final.loc[df_final['Team Member'].notna(), 'Status'] = ''
            
            df_final.loc[df_final['Sprint'] == 'Grand Total', 'Status'] = None
            # print(df_final)
            
            sprint_details = df_final.to_json(orient='records')

            df2_copy = df2.copy()
            df2_copy['Actual Start Date'] = pd.to_datetime(df2_copy['Actual Start Date'], errors='coerce')
            df2_copy['Actual End Date'] = pd.to_datetime(df2_copy['Actual End Date'], errors='coerce')
            
            df_status = pd.DataFrame()
            df_status['Sprint_Start'] = df2_copy.groupby('Sprint')['Actual Start Date'].min()
            df_status['Sprint_End'] = df2_copy.groupby('Sprint')['Actual End Date'].max()
            
            sprint_status = df_status.to_json(orient='records')


            df_pbi = pd.DataFrame()
            df_pbi['Sum of Estimated Effort (Hours)'] = df2.groupby(['Sprint','Team Member','Deliverable'])['Estimated Effort (Hours)'].sum()
            df_pbi['Sum of Actual Effort Hours'] = df2.groupby(['Sprint','Team Member','Deliverable'])['Actual Effort Hours'].sum()
            df_pbi['Sum of ETC (Estimate to Complete)'] = df2.groupby(['Sprint','Team Member','Deliverable'])['ETC (Estimate to Complete)'].sum()
            df_pbi = df_pbi.reset_index()

            df_pbi2 = pd.DataFrame()
            df_pbi2['Sum of Estimated Effort (Hours)'] = df2.groupby('Sprint')['Estimated Effort (Hours)'].sum()
            df_pbi2['Sum of Actual Effort Hours'] = df2.groupby('Sprint')['Actual Effort Hours'].sum()
            df_pbi2['Sum of ETC (Estimate to Complete)'] = df2.groupby('Sprint')['ETC (Estimate to Complete)'].sum()
            df_pbi2 = df_pbi2.reset_index()
            df_pbi2['Deliverable'] = None
            df_pbi2['Team Member'] = None

            df_pbi_concat = pd.concat([df_pbi, df_pbi2], ignore_index=True)


            df_pbi_sorted = df_pbi_concat.sort_values(by=['Sprint','Team Member', 'Deliverable'], ascending=[True,True,True])

            grand_total2 = df_pbi2.drop(columns=['Sprint','Team Member','Deliverable']).sum()

            grand_total_row2 = pd.DataFrame({
                'Sprint': ['Grand Total'],
                'Team Member': [None],
                'Deliverable': [None], 
                **grand_total2.to_dict()
                })

            df_pbi_final = pd.concat([df_pbi_sorted, grand_total_row2], ignore_index=True)
            
            pbi_details = df_pbi_final.to_json(orient='records')

            user_request = question

            output_parser = StrOutputParser()
            router_chain = Router | self.config.llm3 | output_parser
            rounter_request = router_chain.invoke({'user_request':user_request})
            print(rounter_request)
            
            Sprint_Utilization_Agent = Agent(role="Sprint Utilization Agent",
                                goal="Analyze team member workload by comparing capacity with estimated effort and identifying any over or underutilization risks for each sprint.",
                                backstory=("You are a precision-driven analyst dedicated to ensuring sprint efficiency and balanced workloads within a team."),
                                allow_delegation=False,
                                verbose=True,
                                llm= self.config.llm4)
            
            # Sprint_Utilization_Agent.step_callback = PrintThoughtLogger()

            Team_Member_Progress_Agent = Agent(role="Team Member Progress Agent",
                                        goal="Evaluate whether planned sprint work can be completed by analyzing estimated effort, actual effort, and estimate to complete for each team member.",
                                        backstory=(
                                            "You are a progress-tracking expert focused on sprint-level execution. "
                                            "Your responsibility is to monitor how work is progressing across the team members."
                                            "Your insights are grouped by team member and help project managers quickly identify if team members are on track, ahead, or falling behind. "
                                            "You provide clear, actionable visibility into team members progress within a sprint to enable informed decision-making and timely interventions."
                                        ),
                                        allow_delegation=False,
                                        verbose=True,
                                        llm= self.config.llm4)
            
            
            PBI_Progress_Agent = Agent( role="PBI Progress Agent",
                                goal="Track the progress of Product Backlog Items (PBIs) by analyzing estimated effort, actual effort, and remaining effort to determine if work is on track for completion.",
                                backstory=(
                                    "You are a detail-oriented analyst specializing in monitoring the progress of PBIs during a sprint. "
                                    "Your responsibility is to evaluate each PBI assigned to team members"
                                    "Your reports are grouped at the PBI level and help project managers understand the true status of individual work items—identifying risks, delays, or potential overruns early. "
                                    "You provide precise, actionable insights to support sprint delivery and improve sprint planning accuracy."
                                ),
                                allow_delegation=False,
                                verbose=True,
                                llm= self.config.llm4)
            

            su_plotting_Agent = Agent(
                        role="Insightful Plot Generator",
                        goal="Analyze a dataset, generate one observation at a time with corresponding Seaborn plot code, and return both",
                        backstory=(
                            "You are a data visualization specialist. Given a dataset, your job is to generate one insight at a time "
                            "along with Python Seaborn code that visualizes that insight clearly."
                        ),
                        allow_delegation=False,
                        verbose=True,
                        llm=self.config.llm4)
            
            tm_plotting_Agent = Agent(
                        role="Insightful Plot Generator",
                        goal="Analyze a dataset, generate one observation at a time with corresponding Seaborn plot code, and return both",
                        backstory=(
                            "You are a data visualization specialist. Given a dataset, your job is to generate one insight at a time "
                            "along with Python Seaborn code that visualizes that insight clearly."
                        ),
                        allow_delegation=False,
                        verbose=True,
                        llm=self.config.llm4)
            
            # Plotting_Agent.step_callback = PrintThoughtLogger()


            sprint_utilization_task = Task(
                                        description=(
                                            "Analyze the {sprint_details} and {user_request} then generate a **Sprint Utilization Report** as per below instructions" 
                                            "For Example: 'What is Alice's utilization for Sprint 2 and 7 ?'What is Alice's utilization for Sprint 2 and 7? It indicates that the user is requesting a Sprint Utilization Report specifically for Alice(no collaboration), limited to Sprints 2 and 7, excluding all other sprints and team members. "
                                            "For calculating Planning Variance(%) = ((Sum of Capacity - Sum of Estimated Effort)/Sum of Capacity)*100"
                                            "For calculating Sprint End Variance(%) = ((Sum of Actual Effort - Sum of Estimated Effort)/Sum of Estimated Effort)*100"
                                            "For each sprint, list the following per team member:Team Member Name,Sum of Capacity, Sum of Estimated Effort, Sum of Actual Effort, Planning Variance, and Sprint End Variance"
                                            "Also, include a **total row** for each sprint by adding Sum of Capacity,Sum of Estimated Effort, Sum of Actual Effort, Planning Variance(%) and Sprint End Variance(%) of all team members per sprint "
                                            "Additionally, **carefully highlight all the team members whose planning variance exceeds ±10 percent of their capacity."
                                            "Additionally, **carefully highlight all the team members whose sprint end variance exceeds ±10 percent of their estimated effort."
                                            "Carefully highlight planning variance and sprint end variance for all the sprint totals that exceeds ±10 percent."
                                            "Ensure to carefully highlight planning variance and sprint end variance exceeding ±10 percent for each record in the table. Don't skip any records."
                                            "Ensure that when you are highlighting variance exceeding ±10 percent of capacity, always indicate in the variance column."
                                            "Ensure each sprint is grouped clearly and data is structured in a clean, readable table matching this format."
                                            "Ensure that team members working in collaboration with each other are also included in the report."
                                            "Ensure that the output table is sorted in ascending order of Sprint number."
                                            "Ensure that the status for each sprint is properly mentioned in the sprint totals row."
                                            "If the user asks for analysis on all sprints, then consider all sprints like sprint names that have numbers,sprint with weired names like 'Sprint X' 'Deprioritized',sprint with dates like '2025 Q1 Sprint 1' etc."
                                       ),
                                        expected_output=(
                                            "A formatted sprint-level utilization table showing each team member's Capacity vs Estimated Effort vs Actual Effort vs Planning Variance vs Sprint End Variance grouped by sprints,along with sprint-level totals, sprint completion status and highlights for planning variance and sprint end variance deviations exceeding ±10%."
                                            "Do not provide any key findings or any additional analysis/ comments. Only provide the analysis table."
                                            "Along with the table provide a to the point, well structured response clearly answering to user's request."
                                            "All pipes (|), all headings etc should be properly aligned in the output table. They follow proper straight lines and should not run here and there. "
                                            "There are only 6 column headers in the output table: Sum of Capacity, Sum of Estimated Effort, Sum of Actual Effort, Planning Variance, Sprint End Variance and Status."
                                            "Do not add any column header for team member column, leave it blank."
                                            "Do not include Team Member header in the output table."
                                            "Do not include Sprint header in the output table."
                                            "Ensure column headers are aligned properly."
                                            "Do not include multiple tables in the output. Only one table is required."
                                            "You are provided with a sample for reference:"
                                            "**Below is the Output Format Example you need to follow:**"
                                            '''
                                            |               | Sum of Capacity | Sum of Estimated Effort|Sum of Actual Effort|      Planning Variance(%)      |      Sprint End Variance(%)   |   Status     |
                                            | Hariharan     |    64           |        56              |     60             |               13% !            |               7%              |              |
                                            | Raj           |    40           |        48              |     48             |              -20% !            |               0               |              |
                                            | Sprint 1 Total|   104           |       104              |     0              |               0                |               0               | Completed    |
                                            ..............................................................................................................................................................
                                            | Swetha        |    40           |        38              |     38             |               5%               |               0               |              |
                                            | Thooyavan     |    80           |        58              |     58             |               27% !            |               0               |              |
                                            | Sprint 2 Total|    120          |        96              |     96             |               20% !            |               0               |  In Progress |
                                            | Grand Total   |    224          |       200              |     96             |               10%              |               52% !           |              |
                                            
                                            '''
                                        ),
                                        output_parser= StrOutputParser(),
                                        output_file='initial_analysis_report.md',
                                        agent=Sprint_Utilization_Agent)
            

            team_member_progress_task = Task(
                                            description=(
                                                "Analyze the {sprint_details} and {user_request} then generate a **Team Member Progress Report** that evaluates if the planned work assigned to each team member per sprint can be completed on time."
                                                "For Example : 'What is Alice's progress ?' Then it's understood that user is asking to provide Team Member Progress report only for Alice for all sprints. Therefore consider only those records where only and only Alice is present and not in collaboration with any other team member."
                                                "For Example : 'What is Alice's and Mat's progress ?' Then it's understood that user is asking to provide Team Member Progress report for Alice and Mat separately for all sprints. Therefore consider only individual records of Alice and Mat and not in collaboration with each other or any other team member."
                                                "For example: 'Highlight where varaince is more than ±10 hours between the estimated effort and the estimate at completion by team members.' This clearly indicates that the user wants to identify all team members in the progress report whose variance falls outside the ±10 range."
                                                "For calculating Planning Variance(%) = ((Sum of Capacity - Sum of Estimated Effort)/Sum of Capacity)*100"
                                                "For calculating Sprint End Variance(%) = ((Sum of Actual Effort - Sum of Estimated Effort)/Sum of Estimated Effort)*100"
                                                "For **each team member**, report the following for every sprint:Team Member Name,Sum of Estimated Effort,Sum of Actual Effort,Sum of Estimate to Complete (ETC), Sum of Estimate at Completion (EAC = Actual + ETC),Planning Variance(%), and Sprint End Variance(%)"
                                                "Also, include a **total row** for each sprint by adding Sum of Capacity,Sum of Estimated Effort, Sum of Actual Effort, Planning Variance(%) and Sprint End Variance(%) of all team members per sprint "
                                                "Additionally, **carefully highlight all the team members whose planning variance exceeds ±10 percent of their capacity."
                                                "Additionally, **carefully highlight all the team members whose sprint end variance exceeds ±10 percent of their estimated effort."
                                                "Carefully highlight planning variance and sprint end variance for all the sprint totals that exceeds ±10 percent."
                                                "Ensure to carefully highlight planning variance and sprint end variance exceeding ±10 percent for each record in the table. Don't skip any records."
                                                "Ensure that when you are highlighting variance exceeding ±10 percent of capacity, always indicate in the variance column."
                                                "Ensure each sprint is grouped clearly and data is structured in a clean, readable table matching this format."
                                                "Ensure that team members working in collaboration with each other are also included in the report."
                                                "Ensure that the output table is sorted in ascending order of Sprint number."
                                                "Ensure that the status for each sprint is properly mentioned in the sprint totals row."
                                                "If the user asks for analysis on all sprints, then consider all sprints like sprint names that have numbers,sprint with weired names like 'Sprint X' 'Deprioritized',sprint with dates like '2025 Q1 Sprint 1' etc."
                                            ),
                                            expected_output=(
                                                "A detailed sprint-wise progress table grouped by team member, showing Capacity vs Estimated Effort vs Actual Effort vs ETC vs EAC vs Planning Variance vs Sprint End Variance grouped by sprints,along with sprint-level totals, sprint completion status and highlights for planning variance and sprint end variance deviations exceeding ±10%."
                                                "Along with the table provide a to the point, well structured response clearly answering to user's request."
                                                "All pipes (|), all headings etc should be properly aligned in the output table. They follow proper straight lines and should not run here and there. "
                                                "There are only 8 column headers in the output table: Sum of Capacity, Sum of Estimated Effort, Sum of Actual Effort, Sum of ETC, Sum of EAC, Planning Variance, Sprint End Variance and Status."
                                                "Do not add any column header for team member column, leave it blank."
                                                "Do not include Team Member header in the output table."
                                                "Do not include Sprint header in the output table."
                                                "Ensure column headers are aligned properly."
                                                "Do not include multiple tables in the output. Only one table is required."
                                                "You are provided with a sample for reference:"
                                                "**Below is the Output Format Example you need to follow:**"
                                               '''
                                            |               | Sum of Capacity | Sum of Estimated Effort|Sum of Actual Effort| Sum of ETC   | Sum of EAC   |   Planning Variance(%) |    Sprint End Variance(%) |   Status     |
                                            | Hariharan     |    64           |        56              |     60             |     0        |     60       |       13% !            |           7%              |              |
                                            | Raj           |    40           |        48              |     48             |     0        |     48       |      -20% !            |           0               |              |
                                            | Sprint 1 Total|   104           |       104              |     0              |     0        |     0        |       0                |           0               | Completed    |
                                            ...................................................................................................................................................................................
                                            | Swetha        |    40           |        38              |     38             |     0        |     38       |       5%               |           0               |              |
                                            | Thooyavan     |    80           |        58              |     58             |     0        |     58       |       27% !            |           0               |              |
                                            | Sprint 2 Total|    120          |        96              |     96             |     0        |     96       |       20% !            |           0               |  In Progress |
                                            | Grand Total   |    224          |       200              |     96             |     0        |     96       |       10%              |           52% !           |              |
                                        
                                            '''
                                            ),
                                            output_parser=StrOutputParser(),
                                            output_file='initial_analysis_report.md',
                                            agent=Team_Member_Progress_Agent)


            pbi_progress_task = Task(
                                    description=(
                                        "Analyze the {pbi_details} and {user_request} then generate a **PBI Progress Report** that monitors the status of Product Backlog Items (PBIs) per sprint." 
                                        "For **each PBI**, report the following for every sprint:PBI Name,Sum of Estimated Effort,Sum of Actual Effort,Sum of ETC,Sum of EAC (EAC = Actual + ETC),Sum of Variance(EAC - Planned Effort)"
                                        "The report must be **grouped by Sprint**, and within each sprint, by **PBI**."
                                        "Each sprint section should include a **total row** aggregating Sum of Estimated Effort,Sum of Actual Effort,Sum of ETC,Sum of EAC (EAC = Actual + ETC),Sum of Variance(EAC - Planned Effort) for all PBIs in that sprint"
                                        "Ensure proper alignment of columns for readability and consistency. Provide crisp, actionable visibility into the progress of PBIs to support sprint delivery decisions."
                                        "Important Note: Active PBIs are those that have  Sum of ETC as non zero and closed PBIs are those that have  Sum of ETC as zero"
                                    ),
                                    expected_output=(
                                        "A clearly structured sprint-wise PBI progress table showing planned effort, actual effort, ETC, EAC, and variance per PBI with totals aggregated per sprint. "
                                        "Along with the table provide a to the point, well structured response clearly answering to user's request."
                                        "The output table should always contain the following column, in the exact same order: ['Sprint','Team Member','Deliverable','Sum of Est. Effort','Sum of Actual Effort','Sum of ETC','Sum of EAC','Sum of Var']"
                                        "All dash (-), all pipes (|), all headings etc should be properly aligned in the output table. They follow proper straight lines and should not run here and there. "
                                        "You are provided with a sample for reference:"
                                        "**Output Format Example:**"
                                       '''

                                        |                   | Deliverable                                 | Sum of Est. Effort | Sum of Actual Effort  | Sum of ETC | Sum of EAC | Sum of Var |                                       
                                        | Alice             | Update all Documentation for Backstage      |          56        |         60            |      0     |      60    |     4      |
                                        | Bob               | Backstage Handover Doc Update, Review       |           8        |         8             |      0     |       8    |     0      |
                                        | Sprint 1 Total    |                                             |          64        |         68            |      0     |      68    |     4      |
                                        .......................................................................................................................................................
                                        | Charlie           | Dependency Listing Chart                    |          56        |         57            |      0     |      57    |     1      |
                                        | Diana             | Module-Dependency-Version-Listing POC       |          32        |         34            |      0     |      34    |     2      |
                                        | Sprint 2 Total    |                                             |          88        |         91            |      0     |      91    |     3      |

                                        '''
                                    ),
                                    output_parser=StrOutputParser(),
                                    agent=PBI_Progress_Agent
                                )
            su_plotting_task = Task(
                            description=( "You are an expert Data Analyst. Given a JSON-formatted dataframe {sprint_details} and your job is to:\n"
                                           "1. First, write straightforward pandas code using the DataFrame named 'df' to filter the data as per the user request: {user_request}\n"
                                            "   - Do NOT include code to load, assign, or convert sprint_details into a DataFrame. Assume 'df' is already defined and available.\n"
                                            "2. Then, continue with the plotting code using the filtered dataframe(s).\n"
                                            "3. The final output must include:\n"
                                            "  - Filtering code (using 'df')\n"
                                            "  - Then the plotting code using Seaborn/Matplotlib\n"
                                            "  - Write Seaborn/Matplotlib?Pandas code with only 'df' (no other variable name). Do not define dataframes with any other name other than 'df'. All operations to be performed using 'df' variable only.\n"
                                            "  - For all the plots you are not at all allowed to use if and for blocks in the pandas/seaborn/matplotlib code they cause indentation erros while executing."
                                            "4. DO NOT include the sprint data or data conversion steps (like json.loads or pd.DataFrame()).\n"
                                            "5. Ensure to sort the data by Sprint number in ascending order wherever required.\n"
                                            "6. All graphs must be clean, easy to understand, and uncluttered.\n" 
                                            "7. The visualizations are divided into 3 groups namely Sprint Planning, Sprint Execution, Sprint End. Each group conatins it's own set of plots:\n"
                                            "   A. Sprint Planning Plots: should be plotted only for sprint totals, but only for those sprints where the total (summary) row — i.e., rows with no specific team member — has the Status marked as 'Not Started',sorted by sprint number in x axis. Below are sprint planning plots:  "
                                                    "   a. Total_Capacity_vs_Estimated_Effort_vs_Available_Capacity — save as 'Total_Capacity_vs_Estimated_Effort_vs_Available_Capacity.png'\n"
                                                          "- In Total_Capacity_vs_Estimated_Effort_vs_Available_Capacity calculate Available Capacity = Sum of Capacity - Sum of Estimated Effort"
                                                          "- In Total_Capacity_vs_Estimated_Effort_vs_Available_Capacity chart, somewhere in the bottom provide the following information indicating what is purpose of the chart 'Indicates the available capacity that can be allocated for upcoming sprints.'"
                                            "   B. Sprint Execution Plots: should be plotted only for sprint totals, but only for those sprints where the total (summary) row — i.e., rows with no specific team member — has the Status marked as 'In Progress',sorted by sprint number in x axis. Below are sprint execution plots:  "
                                                    "   a. No plots in this category.\n"
                                             "  C. Sprint End Plots: should be plotted only for sprint totals, but only for those sprints where the total (summary) row — i.e., rows with no specific team member — has the Status marked as 'Completed',sorted by sprint number in x axis. Below are sprint end plots:  "
                                                    "   a. Sprint End Effort Variance(%):  — To be calculated strictly using this formula Sprint End Effort Variance(%) = ((Sum of Actual Effort - Sum of Estimated Effort)/Sum of Estimated Effort)*100- save as 'Sprint_End_Effort_Variance(%).png'\n"
                                                    "      - In Sprint End Effort Variance(%) chart, somewhere in the bottom provide the following information indicating what is purpose of the chart 'Highlights effort overruns or underruns by comparing planned effort with actual effort for completed sprints.'"
                                                    "   b. Sprint End Effort Utilization(%): To be calculated strictly using this formula (Sum of Actual Effort for completed sprints / Sum of Capacity for completed sprints)*100  — save as 'Sprint_End_Effort_Utilization(%).png'\n"
                                                    "      - In Sprint End Effort Utilization(%) chart, somewhere in the bottom provide the following information indicating what is purpose of the chart 'Identifies the lost or unallocated capacity in a sprint, calculated as the difference between actual and planned effort for completed sprints'"
                                            "\n"
                                            "8. Visualization Guidelines:\n"
                                            "   a. Sprint-level Analysis (df[df['Team Member'].isnull()]):\n"
                                            "       All plots except for Sprint Utilization should have sprints sorted in ascending order. "
                                            "      - For Category A plot a: bar plot | X: Sprints  | Y: Hours for Total Capacity, Estimated Effort and Available Capacity | Annotate percentage on each bar by normalizing all three metrics by Total Capacity | Hue: Metric | Legend \n"
                                            "      - For Category C plot a: line plot| X: Sprints  | Y: Sprint End Effort Variance Percentage | Reflines: 0%, +10% for overrun, -10% for underrun and mean(mean to be presented in dark blue dashed line) | Annotate percentage for data point \n"
                                            "      - For Category C plot b: line plot| X: Sprints  | Y: Sprint End Effort Utilization Percentage | Reflines: 100%, +10% for overrun, -10% for underrun and mean(mean to be presented in dark blue dashed line) | Annotate percentage for data point\n"
                                            
                                            "   b. Plot Requirements:\n"
                                            "      - All plots must include clear titles, axis labels, and legends\n"
                                            "      - Reference lines must be clearly labeled in the legend\n"
                                            "      - Use consistent colors across plots for the same metrics\n"
                                            "      - Ensure all text elements are readable and not overlapping\n"
                                           "9. For each visualization:\n"
                                            "   - Provide a clear insight in plain English\n"
                                            "   - Add proper titles, labels, and legends\n"
                                            "   - Set figure size and use plt.tight_layout()\n"
                                            "   - Use side-by-side bar plots with hue (not multiple barplot calls)\n"
                                            "   - Use plt.figure(figsize=(width, height))\n"
                                            "   - Rotate labels: plt.xticks(rotation=45)\n"
                                            "   - Add grid: plt.grid(True, alpha=0.3)\n"
                                            "   - For plots with figtext (explanatory text at bottom):\n"
                                            "      * Place figtext with plt.figtext(0.5, 0.01, text, ha='center', fontsize=12, style='italic')\n"
                                            "      * Call plt.tight_layout() first\n"
                                            "      * Then call plt.subplots_adjust(bottom=0.20) to prevent overlap\n"
                                            "      * Finally call plt.savefig with bbox_inches='tight'\n"
                                            "   - Do NOT use the split() method in the plotting code for metric extraction or string operations.\n"
                                         
                                            "\n"
                                            "   IMPORTANT: Do not use triple quotes (\"\"\") in the code strings.\n"
                                            "   Avoid using for loops in the plot code, they cause alot of indentation errors.\n"
                                            "   Use single quotes (') for strings inside the code.\n"
                                            "   Escape newlines with \\n.\n"
                                            "   - Do NOT return code with indentation errors.\n"
                            ),
                            expected_output=("A JSON string containing:\n"
                                                "1. Do not any comments or explanations or headings or verbose text like 'Looking at the sprint data, I need to analyze where estimated effort is ±10 percent of available capacity and crearocess the data and generate the required visualizations with proper Seaborn code.'\n"
                                                "2. Do not include any text before the code block. Just start with the code block.\n"
                                                "3. Do not wrap the code in any json tags in the code block like ```json .....```\n"
                                                "4. Straight away generate the dictionary for visualization.\n"
                                                "5. 'code': A dictionary mapping visualization names to their corresponding plotting code strings\n"
                                                "6. Each code string must be properly escaped for JSON\n"
                                                "7. No triple quotes allowed\n"
                                                "8. All newlines must be escaped with \\n\n"
                                                "9. Output Format:\n"
                                                    "   Return a JSON string with this structure:\n"
                                                    "   {{\n"
                                                    "     \"code\":{{\n"
                                                    "       \"Total_Capacity_vs_Estimated_Effort_vs_Available_Capacity\": \"plt.figure(figsize=(12, 6))\\nsns.barplot(...)\\nplt.title(...)\\nplt.savefig('images/Total_Capacity_vs_Estimated_Effort_vs_Available_Capacity.png', dpi=300)\",\n"
                                                    "       \"Sprint End Effort Variance(%)\": \"plt.figure(figsize=(12, 6))\\nsns.lineplot(...)\\nplt.axhline(...)\\nplt.title(...)\\nplt.savefig('images/Sprint_End_Effort_Variance(%).png', dpi=300)\",\n"
                                                    "       \"Sprint End Effort Utilization(%)\": \"plt.figure(figsize=(12, 6))\\nsns.lineplot(...)\\nplt.axhline(...)\\nplt.title(...)\\nplt.savefig('images/Sprint_End_Effort_Utilization(%).png', dpi=300)\"\n"
                                                    "     }}\n"
                                                    "   }}\n"
                                                    "\n"
                            ),
                            output_key='plotting_code',
                            output_parser=StrOutputParser(),
                            agent=su_plotting_Agent)
            
            tm_plotting_task = Task(
                            description=( "You are an expert Data Analyst. Given a JSON-formatted dataframe {sprint_details} and your job is to:\n"
                                           "1. First, write straightforward pandas code using the DataFrame named 'df' to filter the data as per the user request: {user_request}\n"
                                            "   - Do NOT include code to load, assign, or convert sprint_details into a DataFrame. Assume 'df' is already defined and available.\n"
                                            "2. Then, continue with the plotting code using the filtered dataframe(s).\n"
                                            "3. The final output must include:\n"
                                            "  - Filtering code (using 'df')\n"
                                            "  - Then the plotting code using Seaborn/Matplotlib\n"
                                            "  - Write Seaborn/Matplotlib?Pandas code with only 'df' (no other variable name). Do not define dataframes with any other name other than 'df'. All operations to be performed using 'df' variable only.\n"
                                            "  - For all the plots you are not at all allowed to use if and for blocks in the pandas/seaborn/matplotlib code they cause indentation erros while executing."
                                            "4. DO NOT include the sprint data or data conversion steps (like json.loads or pd.DataFrame()).\n"
                                            "5. Ensure to sort the data by Sprint number in ascending order wherever required.\n"
                                            "6. All graphs must be clean, easy to understand, and uncluttered.\n" 
                                            "7. The visualizations are divided into 3 groups namely Sprint Planning, Sprint Execution, Sprint End. Each group contains it's own set of plots:\n"
                                            "   A. Sprint Planning Plots: should be plotted for individual team members who worked in a sprint, but only for those sprints where the total (summary) row — i.e., rows with no specific team member has the Status marked as 'Not Started',sorted by sprint number in x axis. Below are sprint planning plots:  "
                                                    "   a. Total_Capacity_vs_Estimated_Effort_vs_Available_Capacity — save as 'Total_Capacity_vs_Estimated_Effort_vs_Available_Capacity.png'\n"
                                                    "         - For each sprint, ensure the plot displays exactly three bars for every team member: Total Capacity, Estimated Effort, and Available Capacity.\n"
                                                    "         - For example, if a sprint has 2 team members, the chart should show a total of 6 bars (2 members × 3 metrics).\n"
                                                    "         - Available Capacity for each team member is calculated as: Available Capacity = Sum of Capacity - Sum of Estimated Effort\n"
                                                    "         - Use a grouped bar plot where each group represents one team member in a sprint, and each group contains the three metrics as bars.\n"
                                                    "         - At the bottom of the chart, include the text: 'Indicates the available capacity that can be allocated for upcoming sprints for each team member.'\n"
                                                    "   b. Distribution_of_allocated_work_across_team_members. - save as 'Distribution_of_allocated_work_across_team_members.png' \n"
                                                    "          - In Distribution_of_allocated_work_across_team_members for each team member is calculated as : Distribution_of_allocated_work_across_team_members = (Sum of Estimated Effort per team member/ Sum of Estimated Effort for complete Sprint)*100\n"
                                                    "          - Use a grouped bar plot where each group represents one team member in a sprint, and each group contains the work distribution percentage per team member in that sprint as bars.\n"
                                            "   B. Sprint Execution Plots: should be plotted for individual team members who worked in a sprint, but only for those sprints where the total (summary) row — i.e., rows with no specific team member has the Status marked as 'In Progress',sorted by sprint number in x axis. Below are sprint execution plots:  "
                                                    "   a. No plots in this category.\n"
                                             "  C. Sprint End Plots: should be plotted for individual team members who worked in a sprint, but only for those sprints where the total (summary) row — i.e., rows with no specific team member has the Status marked as 'Completed',sorted by sprint number in x axis. Below are sprint end plots:  "
                                                    "   a. Sprint End Effort Variance(%):  — To be calculated strictly using this formula Sprint End Effort Variance(%) = ((Sum of Actual Effort - Sum of Estimated Effort)/Sum of Estimated Effort)*100- save as 'Sprint_End_Effort_Variance(%).png'\n"
                                                    "      - In Sprint End Effort Variance(%) chart, somewhere in the bottom provide the following information indicating what is purpose of the chart 'Highlights effort overruns or underruns for each team member by comparing planned effort with actual effort for completed sprints.'"
                                                    "   b. Sprint End Effort Utilization(%): To be calculated strictly using this formula (Sum of Actual Effort for completed sprints / Sum of Capacity for completed sprints)*100  — save as 'Sprint_End_Effort_Utilization(%).png'\n"
                                                    "      - In Sprint End Effort Utilization(%) chart, somewhere in the bottom provide the following information indicating what is purpose of the chart 'Identifies the lost or unallocated capacity in a sprint for each team member, calculated as the difference between actual and planned effort for completed sprints'"
                                                    "   c. Team_Allocation(%)_At_Sprint_End - save as 'Team_Allocation(%)_At_Sprint_End.png' \n"
                                                    "      - In Team_Allocation(%)_At_Sprint_End for each team member is calculated as : Team_Allocation(%)_At_Sprint_End = (Sum of Actual Effort per team member/ Sum of Actual Effort for complete Sprint)*100\n"
                                            "\n"
                                            "8. Visualization Guidelines:\n"
                                            "   a. Sprint-level Analysis (df[df['Team Member'].isnull()]):\n"
                                            "       All plots except for Sprint Utilization should have sprints sorted in ascending order. "
                                            "      - For Category A plot a: bar plot | X: Sprint-Team Member  | Y: Hours for Total Capacity, Estimated Effort and Available Capacity | Annotate percentage on each bar by normalizing all three metrics by Total Capacity | Hue: Metric | Legend \n"
                                            "      - For Category A plot b: bar plot | X: Sprint-Team Member | Y: Work Distribution Percentage | Color for all the bars is orange | Annotate percentage on each bar | Legend \n"
                                            "      - For Category C plot a: line plot| X: Sprints  | Y: Sprint End Effort Variance Percentage for each team member| Reflines: 0%, +10% for overrun, -10% for underrunand mean(mean to be calculated on sprint totals not individual team member and presented in dark blue dashed line) | Legend indicating each team member \n"
                                            "      - For Category C plot b: line plot| X: Sprints  | Y: Sprint End Effort Utilization Percentage for each team member | Reflines: 100%, +10% for overrun, -10% for underrun and mean(mean to be calculated on sprint totals not individual team member and presented in dark blue dashed line) | Legend indicating each team member\n"
                                            "      - For Category C plot c: line plot| X: Sprints  | Y: Team Allocation Percentage for each team member at sprint end | Legend indicating each team member\n"
                                            
                                            "   b. Plot Requirements:\n"
                                            "      - All plots must include clear titles, axis labels, and legends\n"
                                            "      - Reference lines must be clearly labeled in the legend\n"
                                            "      - Use consistent colors across plots for the same metrics\n"
                                            "      - Ensure all text elements are readable and not overlapping\n"
                                           "9. For each visualization:\n"
                                            "   - Provide a clear insight in plain English\n"
                                            "   - Add proper titles, labels, and legends\n"
                                            "   - Set figure size and use plt.tight_layout()\n"
                                            "   - Use side-by-side bar plots with hue (not multiple barplot calls)\n"
                                            "   - Use plt.figure(figsize=(width, height))\n"
                                            "   - Rotate labels: plt.xticks(rotation=45)\n"
                                            "   - Add grid: plt.grid(True, alpha=0.3)\n"
                                            "   - For plots with figtext (explanatory text at bottom):\n"
                                            "      * Place figtext with plt.figtext(0.5, 0.01, text, ha='center', fontsize=12, style='italic')\n"
                                            "      * Call plt.tight_layout() first\n"
                                            "      * Then call plt.subplots_adjust(bottom=0.20) to prevent overlap\n"
                                            "      * Finally call plt.savefig with bbox_inches='tight'\n"
                                            "   - Do NOT use the split() method in the plotting code for metric extraction or string operations.\n"
                                         
                                            "\n"
                                            "   IMPORTANT: Do not use triple quotes (\"\"\") in the code strings.\n"
                                            "   Avoid using for loops in the plot code, they cause alot of indentation errors.\n"
                                            "   Use single quotes (') for strings inside the code.\n"
                                            "   Escape newlines with \\n.\n"
                                            "   - Do NOT return code with indentation errors.\n"
                            ),
                            expected_output=("A JSON string containing:\n"
                                                "1. Do not any comments or explanations or headings or verbose text like 'Looking at the sprint data, I need to analyze where estimated effort is ±10 percent of available capacity and crearocess the data and generate the required visualizations with proper Seaborn code.'\n"
                                                "2. Do not include any text before the code block. Just start with the code block.\n"
                                                "3. Do not wrap the code in any json tags in the code block like ```json .....```\n"
                                                "4. Straight away generate the dictionary for visualization.\n"
                                                "5. 'code': A dictionary mapping visualization names to their corresponding plotting code strings\n"
                                                "6. Each code string must be properly escaped for JSON\n"
                                                "7. No triple quotes allowed\n"
                                                "8. All newlines must be escaped with \\n\n"
                                                "9. Output Format:\n"
                                                    "   Return a JSON string with this structure:\n"
                                                    "   {{\n"
                                                    "     \"code\":{{\n"
                                                    "       \"Total_Capacity_vs_Estimated_Effort_vs_Available_Capacity\": \"plt.figure(figsize=(12, 6))\\nsns.barplot(...)\\nplt.title(...)\\nplt.savefig('images/Total_Capacity_vs_Estimated_Effort_vs_Available_Capacity.png', dpi=300)\",\n"
                                                    "       \"Distribution_of_allocated_work_across_team_members\": \"plt.figure(figsize=(12, 6))\\nsns.barplot(...)\\nplt.title(...)\\nplt.savefig('images/Distribution_of_allocated_work_across_team_members.png', dpi=300)\",\n"
                                                    "       \"Sprint End Effort Variance(%)\": \"plt.figure(figsize=(12, 6))\\nsns.lineplot(...)\\nplt.axhline(...)\\nplt.title(...)\\nplt.savefig('images/Sprint_End_Effort_Variance(%).png', dpi=300)\",\n"
                                                    "       \"Sprint End Effort Utilization(%)\": \"plt.figure(figsize=(12, 6))\\nsns.lineplot(...)\\nplt.axhline(...)\\nplt.title(...)\\nplt.savefig('images/Sprint_End_Effort_Utilization(%).png', dpi=300)\",\n"
                                                    "       \"Team_Allocation(%)_At_Sprint_End\": \"plt.figure(figsize=(12, 6))\\nsns.barplot(...)\\nplt.title(...)\\nplt.savefig('images/Team_Allocation(%)_At_Sprint_End.png', dpi=300)\"\n"
                                                    "     }}\n"
                                                    "   }}\n"
                                                    "\n"
                            ),
                            output_key='tm_plotting_code',
                            output_parser=StrOutputParser(),
                            agent=tm_plotting_Agent)

            read_report_tool = FileReadTool(file_path="initial_analysis_report.md")
            read_images_folder_tool = DirectoryReadTool(directory="images")

            # time.sleep(5)

            compiler = Agent(
                name="compiler",
                role="Data Analyst",
                goal=dedent("""
                    To analyze data from the report and associated visual plots, and 
                    synthesize them into a coherent, insightful markdown report. The goal is to narrate 
                    the sprint story clearly and visually.
                """),
                    backstory=dedent("""
                    You are a highly skilled data analyst trained in analyzing sprint metrics and 
                    agile team performance. With expertise in turning raw data and visualizations into 
                    business-readable insights. You are an expert in synthesizing structured reports 
                    from diverse sources and present them in well-formatted markdown files.
                """),
                tools=[read_report_tool, read_images_folder_tool],
                verbose=True,
                llm=self.config.llm4)

            # compiler.step_callback = PrintThoughtLogger()

            su_compile_task = Task(
                description=dedent("""
                    Combine insights from `initial_analysis_report.md` and plots in the `images` folder into a cohesive markdown report `final_analysis_report.md`.
                    Think like a project manager: organize, explain, and narrate the data in a way that helps decision-making. Tie all visuals, tables, and metrics into a clear analytical story.

                    Guidelines:
                    - First thing in the report needs to the analysis table and well defined key performance metrics.              
                    - Then organize the middle of the report by breaking down into three main segments:
                        1. Sprint Planning : 
                            - In sprint planning segment perform analysis only on sprints where Status is 'Not Started'. No other sprints to be included.
                            - Sprint planning plots plus their explanation need to come under this segment. Plots like 'Total Capacity vs Estimated Effort vs Available Capacity' come under this segment.
                        2. Sprint Execution : 
                            - In sprint execution segment perform analysis only on sprints where Status is 'In Progress'. No other sprints to be included.
                            - Right now there are no plots that come under this segment.
                        3. Sprint End: 
                            - In sprint end segment perform analysis only on sprints where Status is 'Completed'. No other sprints to be included.
                            - Sprint end plots plus their explanation need to come under this segment. Plots like 'Sprint End Utilization','Sprint End Variance' come under this segment.
                    - Always present the analysis table exactly the same way present in `initial_analysis_report.md`, do not add more sprints or metrics to original table. Stick to what is provided, don't modify the table.
                    - Finally in the end of the report provide critical findings and recommendations.
                    - Avoid redundancy by embedding same plots again and again. 
                    - Always keep the tone professional, positive, and engaging. Avoid casual or blunt language.
                    - Make use to appropriate emojis whereever needed.
                    - Include all content from `initial_analysis_report.md`, especially the analysis table containing all numbers and metrics:
                        * Do **not** truncate or modify analysis table.
                        * Format as **aligned ASCII table** within triple backticks ```bash.
                    - Explain **all metrics and formulas** used (e.g., Sprint Utilization = Estimated Effort / Capacity * 100).
                    - Include and embed **only** the images from the `images` folder. Do not modify the paths of those images. Keep them as is.
                    - If you're adding new content to the report, do not reference or embed images that were not read by the tool and don't exist in images folder. This causes broken image links in the final report. Avoid such issues.       
                    - Do **not** include any other plots like Team_Utilization_Heatmap or Workload_Distribution from your end, stick to the plots that are provided to you.
                    - For each image:
                        * Embed using `![](images/filename.png)`
                        * Provide detailed interpretation: explain what the plot shows, who is over/under-utilized, and what decisions can be made.
                    - Ensure the number of images matches exactly the number present in the `images` folder.
                    - Format the report in **GitHub-style markdown**, like a polished README.
                    - Do **not** include Introduction, Executive Summary, or Conclusion.
                    - Ensure there are no gaps — connect all text, tables, and visuals into a seamless narrative.
                """),
                expected_output=dedent("""
                    A markdown report saved as "final_analysis_report.md" that includes text, tables  
                    and embedded images exactly as they are in the `images` folder with a cohesive data story. Complete analysis table from 'initial_analysis_report.md' must be present in the report.
                    Number of images in the report should be exactly the same as the number of images present in the `images` folder. 
                    Do not add any new or unnecessary images in the report that are not present in the `images` folder.
                    The file must be formatted like a GitHub README, well-structured, and easy to read.
                """),
                output_file="final_analysis_report.md",
                agent=compiler)
            
            tm_compile_task = Task(
                description=dedent("""
                    Combine insights from `initial_analysis_report.md` and plots in the `images` folder into a cohesive markdown report `final_analysis_report.md`.
                    Think like a project manager: organize, explain, and narrate the data in a way that helps decision-making. Tie all visuals, tables, and metrics into a clear analytical story.

                    Guidelines:
                    - First thing in the report needs to the analysis table and well defined key performance metrics.              
                    - Then organize the middle of the report by breaking down into three main segments:
                        1. Sprint Planning : 
                            - In sprint planning segment perform analysis only on sprints where Status is 'Not Started'. No other sprints to be included.
                            - Sprint planning plots plus their explanation need to come under this segment. Plots like 'Total Capacity vs Estimated Effort vs Available Capacity', 'Distribution_of_allocated_work_across_team_members' etc come under this segment.
                        2. Sprint Execution : 
                            - In sprint execution segment perform analysis only on sprints where Status is 'In Progress'. No other sprints to be included.
                            - Right now there are no plots that come under this segment.
                        3. Sprint End: 
                            - In sprint end segment perform analysis only on sprints where Status is 'Completed'. No other sprints to be included.
                            - Sprint end plots plus their explanation need to come under this segment. Plots like 'Sprint End Utilization','Sprint End Variance','Team Allocation(%) At Sprint End' etc come under this segment. 
                    - Always present the analysis table exactly the same way present in `initial_analysis_report.md`, do not add more sprints or metrics to original table. Stick to what is provided, don't modify the table.
                    - Finally in the end of the report provide critical findings and recommendations.
                    - Avoid redundancy by embedding same plots again and again. 
                    - Always keep the tone professional, positive, and engaging. Avoid casual or blunt language.
                    - Make use to appropriate emojis whereever needed.
                    - Include all content from `initial_analysis_report.md`, especially the analysis table containing all numbers and metrics:
                        * Do **not** truncate or modify analysis table.
                        * Format as **aligned ASCII table** within triple backticks ```bash.
                    - Explain **all metrics and formulas** used (e.g., Sprint Utilization = Estimated Effort / Capacity * 100).
                    - Include and embed **only** the images from the `images` folder. Do not modify the paths of those images. Keep them as is.
                    - If you're adding new content to the report, do not reference or embed images that were not read by the tool and don't exist in images folder. This causes broken image links in the final report. Avoid such issues.       
                    - Do **not** include any other plots like Team_Utilization_Heatmap or Workload_Distribution from your end, stick to the plots that are provided to you.
                    - For each image:
                        * Embed using `![](images/filename.png)`
                        * Provide detailed interpretation: explain what the plot shows, who is over/under-utilized, and what decisions can be made.
                    - Ensure the number of images matches exactly the number present in the `images` folder.
                    - Format the report in **GitHub-style markdown**, like a polished README.
                    - Do **not** include Introduction, Executive Summary, or Conclusion.
                    - Ensure there are no gaps — connect all text, tables, and visuals into a seamless narrative.
                """),
                expected_output=dedent("""
                    A markdown report saved as "final_analysis_report.md" that includes text, tables  
                    and embedded images exactly as they are in the `images` folder with a cohesive data story. Complete analysis table from 'initial_analysis_report.md' must be present in the report.
                    Number of images in the report should be exactly the same as the number of images present in the `images` folder. 
                    Do not add any new or unnecessary images in the report that are not present in the `images` folder.
                    The file must be formatted like a GitHub README, well-structured, and easy to read.
                """),
                output_file="final_analysis_report.md",
                agent=compiler)


            if rounter_request.lower() == 'sprint utilization':

                crew = Crew(
                        agents=[Sprint_Utilization_Agent, su_plotting_Agent],
                        tasks=[sprint_utilization_task, su_plotting_task],
                        verbose=True,
                        process=Process.sequential)
                
                inputs = {'sprint_details':sprint_details, 'user_request':user_request}
                su_result = crew.kickoff(inputs=inputs)
                print("\n=== Final Response ===")
                print(su_result)
                
                try:
                    # Extract plotting code from CrewOutput
                    plotting_code = json.loads(str(su_result))
                    
                    # Create tool input
                    tool_input = {
                        'code': plotting_code['code'],
                        'complete_data': sprint_details
                    }
                    
                    # Call the plotting tool directly
                    plotting_tool = SeabornPlotGeneratorTool()
                    plot_result = plotting_tool._run(**tool_input)
                    
                    final_result = f"{su_result}\n\nPlot Generation Results:\n{plot_result}"
                    
                except Exception as e:
                    logging.error(f"Error in plot generation: {e}")
                    final_result = f"{su_result}\n\nError generating plots: {str(e)}"


                crew2 = Crew(
                    agents=[compiler],
                    tasks=[su_compile_task],
                    verbose=True
                )

                crew2.kickoff()

                
            elif rounter_request.lower() == 'team member progress':
                crew = Crew(agents=[Team_Member_Progress_Agent, tm_plotting_Agent],
                        tasks=[team_member_progress_task, tm_plotting_task],
                        verbose=True,
                        process=Process.sequential)
                
                inputs = {'sprint_details':sprint_details, 'user_request':user_request}
                tm_result = crew.kickoff(inputs=inputs)
                print(tm_result)
                # final_result = Markdown(result)

                 
                try:
                    # Extract plotting code from CrewOutput
                    plotting_code = json.loads(str(tm_result))
                    
                    # Create tool input
                    tool_input = {
                        'code': plotting_code['code'],
                        'complete_data': sprint_details
                    }
                    
                    # Call the plotting tool directly
                    plotting_tool = SeabornPlotGeneratorTool()
                    plot_result = plotting_tool._run(**tool_input)
                    
                    final_result = f"{tm_result}\n\nPlot Generation Results:\n{plot_result}"
                    
                except Exception as e:
                    logging.error(f"Error in plot generation: {e}")
                    final_result = f"{tm_result}\n\nError generating plots: {str(e)}"


                crew2 = Crew(
                    agents=[compiler],
                    tasks=[tm_compile_task],
                    verbose=True
                )

                crew2.kickoff()
                
            else:

                crew = Crew(
                        agents=[PBI_Progress_Agent],
                        tasks=[pbi_progress_task],
                        verbose=True,
                        process=Process.sequential)
                
                inputs = {'pbi_details':pbi_details, 'user_request':user_request}
                result = crew.kickoff(inputs= inputs)
                print(result)
                # final_result = Markdown(result)                   

            print('Effort Analysis Complete',datetime.now())
            logging.info('Effort Analysis Complete')
            stop = datetime.now()
            print("Total Time Taken For Analysis:", stop-start)

            return "Analysis Complete"

        except Exception as e:
            logging.error(f"Unexpected Error: {e}")
            raise CustomException(e, sys)