from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
from src.logger import logging
from src.exception_handler import CustomException
from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import ast
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import sys
from src.llm_initialize import LLMInitializer
from src.data_fetching import DataFetching
from dataclasses import dataclass
from src.prompts import EffortAnalysisPrompt,EffortAnalysisPrompt2, Router
from langchain_core.output_parsers import StrOutputParser
import numpy as np
from crewai import Agent, Task, Crew, Process
from IPython.display import Markdown


@dataclass
class EffortTrackingConfig:   
    def __init__(self, file_name):
        self.llm_init = LLMInitializer()
        self.llm1 = self.llm_init.initialize_llm(model_num=1)
        self.llm2 = self.llm_init.initialize_llm(model_num=2, temperature=0.0)
        self.llm3 = self.llm_init.initialize_llm(model_num=3, temperature=0.0)
        self.llm4 = self.llm_init.initialize_llm(model_num=4, temperature=0.3)
        self.data_fetching = DataFetching(file_name)
        self.df =  self.data_fetching.data_fetching()

class EffortTracking:
    def __init__(self, file_name):
        self.file_name = file_name
        self.config = EffortTrackingConfig(file_name)

    # @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=60), retry=retry_if_exception_type(Exception))
    # def burn_down_generation(self):
    #     try:
    #         logging.info('Burn Down Generation Started')
    #         print('Burn Down Generation Started', datetime.now())

    #         try:
    #             df = self.config.df
    #             columns = df.columns
    #             # print('DF COLUMNS BEFORE DROPPING:', columns)

    #             df2 = df[['Project / Workstream Name', 'Deliverable', 'Variance', 'EAC (Estimate at Completion)', 'ETC (Estimate to Complete)', 'Actual Effort Hours', 'Actual End Date', 'Actual Start Date', 'Estimated Effort (Hours)', 'Sprint', 'Capacity', 'Team Member']]
    #             # print('DF COLUMNS After DROPPING:', df2.columns)

    #             df3 = pd.DataFrame()
    #             df3['Sum of Capacity'] = df2.groupby(['Sprint','Team Member'])['Capacity'].sum()
    #             df3['Sum of Estimated Effort (Hours)'] = df2.groupby(['Sprint','Team Member'])['Estimated Effort (Hours)'].sum()
    #             df3['Sum of Actual Effort Hours'] = df2.groupby(['Sprint','Team Member'])['Actual Effort Hours'].sum()
    #             df3['Sum of ETC (Estimate to Complete)'] = df2.groupby(['Sprint','Team Member'])['ETC (Estimate to Complete)'].sum()
    #             df3['Sum of EAC (Estimate at Completion)'] = df2.groupby(['Sprint','Team Member'])['EAC (Estimate at Completion)'].sum()
    #             df3['Sum of Variance'] = df2.groupby(['Sprint','Team Member'])['Variance'].sum()
    #             df3 = df3.reset_index()
    #             # print(df3)

    #             df4 = pd.DataFrame()
    #             df4['Sum of Capacity'] = df2.groupby('Sprint')['Capacity'].sum()
    #             df4['Sum of Estimated Effort (Hours)'] = df2.groupby('Sprint')['Estimated Effort (Hours)'].sum()
    #             df4['Sum of Actual Effort Hours'] = df2.groupby('Sprint')['Actual Effort Hours'].sum()
    #             df4['Sum of ETC (Estimate to Complete)'] = df2.groupby('Sprint')['ETC (Estimate to Complete)'].sum()
    #             df4['Sum of EAC (Estimate at Completion)'] = df2.groupby('Sprint')['EAC (Estimate at Completion)'].sum()
    #             df4['Sum of Variance'] = df2.groupby('Sprint')['Variance'].sum()
    #             df4 = df4.reset_index()
    #             df4['Team Member'] = None
    #             # print(df4)

    #             df_concat = pd.concat([df3, df4], ignore_index=True)
    #             # print(df_concat)

    #             df_sorted = df_concat.sort_values(by=['Sprint', 'Team Member'], ascending=[True,True])
    #             # print(df_sorted)

    #             grand_total = df4.drop(columns=['Sprint', 'Team Member']).sum()

    #             grand_total_row = pd.DataFrame({
    #                 'Sprint': ['Grand Total'], 
    #                 'Team Member': [None], 
    #                 **grand_total.to_dict()
    #             })

    #             df_final = pd.concat([df_sorted, grand_total_row], ignore_index=True)
    #             # print(df_final)


    #             print('Burn Down Generation Complete',datetime.now())
    #             logging.info('Burn Down Generation Complete')
    #             return df_final

    #         except Exception as e:
    #             logging.error(f"Burn Generation failed Failed: {e}")
    #             raise CustomException(e, sys)

    #     except Exception as e:
    #         logging.error(f"Unexpected Error: {e}")
    #         raise CustomException(e, sys)
        
    # def effort_analysis(self,report,question):
    def effort_analysis(self,question):
        try:
            print('Effort Analysis Started',datetime.now())
            start = datetime.now()
            logging.info('Effort Analysis Started')

            df = self.config.df
            sprint_details = df.to_json(orient='records')

            user_request = question

            output_parser = StrOutputParser()
            router_chain = Router | self.config.llm1 | output_parser
            rounter_request = router_chain.invoke({'user_request':user_request})
            print(rounter_request)

            # Manager = Agent(
            #                 role="Manager Agent",
            #                 goal="Understand user query and assign it to the appropriate worker agent",
            #                 backstory=(
            #                     "You are a skilled coordinator overseeing a team of three specialized workers. "
            #                     "Each worker performs a unique type of analysis on user-provided project data, "
            #                     "focusing on identifying and minimizing time-related risks that could impact project delivery. "
            #                     "Your expertise lies in interpreting user requests, orchestrating the analysis workflow, "
            #                     "and efficiently delegating tasks to the most appropriate worker based on the nature of the request."
            #                     "You force the workers to follow tasks exactly as they are defined for them."
            #                 ),
            #                 allow_delegation=True,
            #                 verbose=True,
            #                 llm= self.config.llm4,
            #             )
            
            Sprint_Utilization_Agent = Agent(role="Sprint Utilization Agent",
                                goal="Analyze team member workload by comparing capacity with estimated effort and identifying any over or underutilization risks for each sprint.",
                                backstory=("You are a precision-driven analyst dedicated to ensuring sprint efficiency and balanced workloads within a team."),
                                allow_delegation=False,
                                verbose=True,
                                llm= self.config.llm4
                            )
            
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
            
            # Generate a chart based on the tabular data in the result
            Chart_Generation_Agent = Agent(
                role="Chart Generation Agent",
                goal="Generate a visual chart (e.g., bar chart, line chart) based on the tabular data provided in the result.",
                backstory=(
                    "You are a visualization expert skilled in creating clear and insightful charts from tabular data. "
                    "Your responsibility is to transform data into visual formats that enhance understanding and decision-making."
                ),
                allow_delegation=False,
                verbose=True,
                llm=self.config.llm4
            )

            chart_generation_task = Task(
                description=(
                    "Using the tabular data provided in the result, generate a **visual chart** that clearly represents the data. "
                    "Ensure the chart type is appropriate for the data (e.g., bar chart for comparisons, line chart for trends). "
                    "Label axes, include a legend if necessary, and ensure the chart is clean and easy to interpret."
                ),
                expected_output=(
                    "A visual chart (e.g., bar chart, line chart) that accurately represents the tabular data provided in the result."
                ),
                agent=Chart_Generation_Agent
            )

            Chart_Image_Generation_Agent = Agent(
                role="Chart Image Generation Agent",
                goal="Convert the chart code (e.g., mermaid or other formats) into an actual image file or displayable graph.",
                backstory=(
                    "You are an expert in rendering visualizations into tangible image formats. "
                    "Your role is to take chart code and produce a high-quality, displayable image or graph."
                ),
                allow_delegation=False,
                verbose=True,
                llm=self.config.llm4
            )

            chart_image_generation_task = Task(
                description=(
                    "Take the chart code (e.g., mermaid or other formats) generated by the Chart Generation Agent and render it into an actual image or graph. "
                    "Ensure the image is clear, properly formatted, and ready for display or embedding."
                ),
                expected_output=(
                    "A rendered image or graph based on the chart code provided."
                ),
                agent=Chart_Image_Generation_Agent
            )

            
            # final_result = Markdown(result)

            sprint_utilization_task = Task(
                                        description=(
                                            "Analyze the {sprint_details} and {user_request} then generate a **Sprint Utilization Report** that compares each team member's Capacity vs Estimated Effort vs Variance grouped by sprints." 
                                            "For calculating Variance = Sum of Capacity - Sum of Estimated Estimated Effort"
                                            "For each sprint, list the following per team member:Team Member Name,Sum of Capacity, Sum of Planned Effort (Estimated Effort) and Variance (Sum of Capacity - Sum of Estimated Estimated Effort)"
                                            "Also, include a **total row** for each sprint by adding Sum of Capacity,Sum of Estimated Effort and Variance of all team members per sprint "
                                            "Additionally, **highlight any team member whose variance exceeds ±10%** of their capacity. "
                                            "This helps determine over-utilization or under-utilization."
                                            "Ensure each sprint is grouped clearly and data is structured in a clean, readable table matching this format."
                                        ),
                                        expected_output=(
                                            "A formatted sprint-level utilization table showing each team member's Capacity vs Estimated Effort vs Variance grouped by sprints, "
                                            "along with sprint-level totals and highlights for variance deviations exceeding ±10%. You are provided with a sample for reference:"
                                            "**Output Format Example:**"
                                            '''
                                            -------------------------------------------------------------------------
                                            |               | Sum of Capacity | Sum of Estimated Effort | Variance  |
                                            -------------------------------------------------------------------------
                                            | Hariharan     |    64           |        56               |     8     |
                                            | Raj           |    40           |        48               |    -8     |
                                            | Sprint 1 Total|   104           |       104               |     0     |
                                            -------------------------------------------------------------------------
                                            '''
                                        ),
                                        agent=Sprint_Utilization_Agent
                                    )
            
            team_member_progress_task = Task(
                                            description=(
                                                "Analyze the {sprint_details} and {user_request} then generate a **Team Member Progress Report** that evaluates if the planned work assigned to each team member per sprint can be completed on time."
                                                "For **each team member**, report the following for every sprint:Team Member Name,Sum of Estimated Effort,Sum of Actual Effort,Sum of Estimate to Complete (ETC), Sum of Estimate at Completion (EAC = Actual + ETC),Sum of Variance (EAC - Estimated Effort)"
                                                "Group by Sprint, then by Team Member, and must include a **total row** per sprint showing: Sum of Estimated Effort,Sum of Actual Effort,Sum of Estimate to Complete (ETC), Sum of Estimate at Completion,Sum of Variance all team members per sprint." 
                                                "Ensure the formatting is clean, aligned, and easy to read. Highlight any variances where applicable for easier insights."
                                            ),
                                            expected_output=(
                                                "A detailed sprint-wise progress table grouped by team member, showing estimated effort, actual effort, ETC, EAC, and variance with totals for each sprint.You are provided with a sample for reference:"
                                                 "Output Format Example:"
                                               ''' 
                                                ---------------------------------------------------------------------------------------------------
                                                |               | Sum of Est. Effort |Sum of Actual Effort | Sum of ETC | Sum of EAC | Sum of Var |
                                                ---------------------------------------------------------------------------------------------------
                                                | Hariharan     |     56             |      60             |  0         | 60         |  4         |
                                                | Raj           |      8             |       8             |  0         |  8         |  0         |
                                                | Sprint 1 Total|     64             |      68             |  0         | 68         |  4         |
                                                ---------------------------------------------------------------------------------------------------
                                                | Hariharan     |     56             |      57             |  0         | 57         |  1         |
                                                | Raj           |     32             |      34             |  0         | 34         |  2         |
                                                |Sprint 2 Total |     88             |      91             |  0         | 91         |  3         |
                                                ---------------------------------------------------------------------------------------------------
                                                '''
                                            ),
                                            agent=Team_Member_Progress_Agent
                                        )

            pbi_progress_task = Task(
                                    description=(
                                        "Analyze the {sprint_details} and {user_request} then generate a **PBI Progress Report** that monitors the status of Product Backlog Items (PBIs) per sprint." 
                                        "For **each PBI**, report the following for every sprint:PBI Name,Sum of Estimated Effort,Sum of Actual Effort,Sum of ETC,Sum of EAC (EAC = Actual + ETC),Sum of Variance(EAC - Planned Effort)"
                                        "The report must be **grouped by Sprint**, and within each sprint, by **PBI**."
                                        "Each sprint section should include a **total row** aggregating Sum of Estimated Effort,Sum of Actual Effort,Sum of ETC,Sum of EAC (EAC = Actual + ETC),Sum of Variance(EAC - Planned Effort) for all PBIs in that sprint"
                                        "Ensure proper alignment of columns for readability and consistency. Provide crisp, actionable visibility into the progress of PBIs to support sprint delivery decisions."
                                    ),
                                    expected_output=(
                                        "A clearly structured sprint-wise PBI progress table showing planned effort, actual effort, ETC, EAC, and variance per PBI with totals aggregated per sprint. You are provided with a sample for reference:"
                                        "**Output Format Example:**"
                                       '''
                                        ----------------------------------------------------------------------------------------------------------------------------
                                        |                                        | Sum of Est. Effort |Sum of Actual Effort | Sum of ETC | Sum of EAC | Sum of Var |
                                        ----------------------------------------------------------------------------------------------------------------------------
                                        | Update all Documentation for Backstage |          56        |         60          |      0     |      60    |     4      |
                                        | Backstage Handover Doc Update, Review  |           8        |         8           |      0     |       8    |     0      |
                                        | Sprint 1 Total                         |          64        |         68          |      0     |      68    |     4      |
                                        ----------------------------------------------------------------------------------------------------------------------------
                                        | Dependency Listing Chart               |          56        |         57          |      0     |      57    |     1      |
                                        | Module-Dependency-Version-Listing POC  |          32        |         34          |      0     |      34    |     2      |
                                        ----------------------------------------------------------------------------------------------------------------------------
                                        | Sprint 2 Total                         |          88        |         91          |      0     |      91    |     3      |
                                        ----------------------------------------------------------------------------------------------------------------------------
                                        '''
                                    ),
                                    agent=PBI_Progress_Agent
                                )
            
            # manager_task = Task(
            #                     description=(
            #                          """Analyze the {user_query} and determine which report is being requested: 
            #                         Sprint Utilization, Team Member Progress, or PBI Progress.

            #                         Follow these steps:
            #                         1. Interpret the user's query to identify the type of analysis or report needed.
            #                         2. Match the query to one of the following specialized worker agents:
            #                         - Sprint Utilization Agent: Handles capacity vs planned effort analysis to identify overloading or underutilization.
            #                         - Sprint Progress Agent: Tracks team member-level progress by comparing planned effort, actual effort, and estimate to complete (ETC).
            #                         - PBI Progress Tracker Agent: Focuses on PBI-level tracking, calculating estimated vs actual effort, ETC, Estimate at Completion (EAC), and variance.
            #                         3. Once the appropriate agent is identified, delegate the query to that agent. Forward {sprint_details} data for analysis and instruct it to execute the corresponding task precisely on the sprint details:
            #                         - sprint_utilization_task
            #                         - team_member_progress_task
            #                         - pbi_progress_task
            #                         4. If the query does not clearly match any known analysis types, request clarification from the user.
            #                         """
            #                     ),
            #                     expected_output=(
            #                         "Delegate the query to the appropriate agent and instruct it to execute the relevant task exactly "
            #                         "(sprint_utilization_task, team_member_progress_task, or pbi_progress_task)."
            #                     ),
            #                     agent=Manager,
            #                     context=[sprint_utilization_task, team_member_progress_task,pbi_progress_task]
            # 
            # 
            #                 )
            
            if rounter_request.lower() == 'sprint utilization':
                crew = Crew(
                        agents=[Sprint_Utilization_Agent],
                        tasks=[sprint_utilization_task],
                        verbose=True,
                        process=Process.sequential
                    )
                inputs = {'sprint_details':sprint_details, 'user_request':user_request}
                result = crew.kickoff(inputs= inputs)
                print(result)
                # final_result = Markdown(result)

            elif rounter_request.lower() == 'team member progress':
                crew = Crew(
                        agents=[Team_Member_Progress_Agent],
                        tasks=[team_member_progress_task],
                        verbose=True
                    )
                inputs = {'sprint_details':sprint_details, 'user_request':user_request}
                result = crew.kickoff(inputs=inputs)
                print(result)


            else:
                crew = Crew(
                        agents=[PBI_Progress_Agent],
                        tasks=[pbi_progress_task],
                        verbose=True
                    )
                inputs = {'sprint_details':sprint_details, 'user_request':user_request}
                result = crew.kickoff(inputs= inputs)
                print(result)
                # final_result = Markdown(result)
                
                    

            print('Effort Analysis Complete',datetime.now())
            logging.info('Effort Analysis Complete')
            stop = datetime.now()
            print("Total Time Taken For Analysis:", stop-start)

            return result
        

        except Exception as e:
            logging.error(f"Unexpected Error: {e}")
            raise CustomException(e, sys)