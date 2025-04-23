from sqlalchemy import create_engine, exc
import os
import pandas as pd
from src.logger import logging
from src.exception_handler import CustomException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import sys
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DataIngestionConfig:
    
    def __init__(self, file_name):
        self.db_path: str = 'sqlite:///database/sqlite_db.db'
        self.excel_file_path = f"data/{file_name}.xlsx"

class DataIngestion:
    def __init__(self, file_name):
        self.ingestion_config = DataIngestionConfig(file_name)
        self.file_name = file_name

    # @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=60), retry=retry_if_exception_type(Exception))
    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")
        try:
            engine = create_engine(self.ingestion_config.db_path)
            excel_data = pd.ExcelFile(self.ingestion_config.excel_file_path)
            sheet_name = excel_data.sheet_names[0]
            df = pd.read_excel(self.ingestion_config.excel_file_path, sheet_name=sheet_name)
            df.columns = df.columns.str.replace('\xa0', ' ').str.strip()
            df.to_sql(self.file_name, con=engine, index=False, if_exists='replace', method='multi')
            # print(df)
            logging.info(f'Ingested {len(df)} rows from sheet: {sheet_name} at {datetime.now()}')

            logging.info("Database ingestion complete")
            return engine
        
        except (FileNotFoundError, exc.SQLAlchemyError) as e:
            raise CustomException(f"Error during data ingestion: {e}", sys)
        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == "__main__":
#     data_ingestion = DataIngestion()
#     engine = data_ingestion.initiate_data_ingestion()

#     llm_init = LLMInitializer()
#     llm = llm_init.initialize_llm(model_num=2, temperature=0.3, top_p=0.9, top_k=50)

#     project_name = 'cep orchestration'
#     data_fetching = DataFetching()
#     fetched_data = data_fetching.fetch_data(engine, project_name)



