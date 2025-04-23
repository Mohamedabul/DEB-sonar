from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
from src.logger import logging
from src.exception_handler import CustomException
from datetime import datetime
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.tools.sql_database.tool import SQLDatabase
# from langchain.chains.sql_database.query import create_sql_query_chain
# from langchain.prompts import PromptTemplate
# import ast
# from langchain.prompts import ChatPromptTemplate
import sys
from src.data_ingestion import DataIngestion
from dataclasses import dataclass
from sqlalchemy import inspect

@dataclass
class DataFetchingConfig:  
    def __init__(self, file_name):
        self.data_ingestion = DataIngestion(file_name)
        self.engine =  self.data_ingestion.initiate_data_ingestion()


class DataFetching:
    def __init__(self, file_name):
        self.file_name = file_name
        self.config = DataFetchingConfig(file_name)

    # @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=60), retry=retry_if_exception_type(Exception))
    def data_fetching(self):
        try:
            logging.info('Data Fetching Started')
            print('Data Fetching Started', datetime.now())     
            
            try:
                with self.config.engine.connect() as conn:
                    inspector = inspect(self.config.engine)
                    tables = [t.replace(" ", "_") for t in inspector.get_table_names()]
                    print(tables)

                    query = f"SELECT * FROM '{self.file_name}'"
                    df = pd.read_sql(query, conn)
                    result = df.to_dict(orient="records")
                    
                # print("RESULT:", result)
                print('QUERY RESULT run time', datetime.now())
            except Exception as e:
                raise CustomException(e, sys)

            print('Data Fetching Complete')
            logging.info('Data Fetching Complete')
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
        


# if __name__ == "__main__":
#     data_fetching = DataFetching('orchestration')
#     df = data_fetching.data_fetching()
#     print(df)