from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from src.logger import logging
from src.exception_handler import CustomException
import sys
from dataclasses import dataclass
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from crewai import LLM

class LLMInitializer:
    def __init__(self):
        self.llm = None

    # @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=60), retry=retry_if_exception_type(Exception))
    def initialize_llm(self,model_num = 1,temperature=0.0,top_p=None, top_k=None):
        try:
            logging.info("---->Model Initialization Begun<-----")
            print("Model Initialization Begun: ", datetime.now())
            load_dotenv()

            AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
            os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY

            AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")  
            os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_KEY

            AWS_REGION = os.getenv("AWS_REGION")
            os.environ["AWS_REGION"] = AWS_REGION

            os.environ["MODEL_ID_1"] = os.getenv("MODEL_ID_1")
            MODEL_ID_1 = os.environ["MODEL_ID_1"]

            os.environ["MODEL_PROVIDER_1"] = os.getenv("MODEL_PROVIDER_1")
            MODEL_PROVIDER_1 = os.environ["MODEL_PROVIDER_1"]

            os.environ["MODEL_ID_2"] = os.getenv("MODEL_ID_2")
            MODEL_ID_2 = os.environ["MODEL_ID_2"]

            os.environ["MODEL_PROVIDER_2"] = os.getenv("MODEL_PROVIDER_2")
            MODEL_PROVIDER_2 = os.environ["MODEL_PROVIDER_2"]

            os.environ["MODEL_ID_3"] = os.getenv("MODEL_ID_3")
            MODEL_ID_3 = os.environ["MODEL_ID_3"]

            os.environ["MODEL_PROVIDER_3"] = os.getenv("MODEL_PROVIDER_3")
            MODEL_PROVIDER_3 = os.environ["MODEL_PROVIDER_3"]

            os.environ["MODEL_ID_4"] = os.getenv("MODEL_ID_4")
            MODEL_ID_4 = os.environ["MODEL_ID_4"]

            os.environ["MODEL_PROVIDER_4"] = os.getenv("MODEL_PROVIDER_4")
            MODEL_PROVIDER_4 = os.environ["MODEL_PROVIDER_4"]


            if model_num == 1:
                model_kwargs = {"max_gen_len": 512, "temperature": temperature}
                if top_p is not None:
                    model_kwargs["top_p"] = top_p
                if top_k is not None:
                    model_kwargs["top_k"] = top_k
                
                self.llm = ChatBedrock(
                    provider=MODEL_PROVIDER_1,
                    model_id=MODEL_ID_1,
                    region_name=AWS_REGION,
                    model_kwargs=model_kwargs,
                    streaming=True,
                    cache=False
                )

                self.llm.provider_stop_sequence_key_name_map = {'anthropic': 'stop_sequences', 'amazon': 'stopSequences',
                'ai21': 'stop_sequences', 'cohere': 'stop_sequences',
                'mistral': 'stop','meta': 'stop'}

            elif model_num == 2:
                model_kwargs = {"temperature": temperature, "max_tokens": 8192}
                if top_p is not None:
                    model_kwargs["top_p"] = top_p
                if top_k is not None:
                    model_kwargs["top_k"] = top_k
                
                self.llm = ChatBedrock(
                    provider=MODEL_PROVIDER_2,
                    model_id=MODEL_ID_2,
                    region_name=AWS_REGION,
                    model_kwargs=model_kwargs,
                    streaming=False,
                    cache=False
                )

                self.llm.provider_stop_sequence_key_name_map = {'anthropic': 'stop_sequences', 'amazon': 'stopSequences',
                'ai21': 'stop_sequences', 'cohere': 'stop_sequences',
                'mistral': 'stop','meta': 'stop'}

            elif model_num ==3: 
                model_kwargs = {
                "temperature": temperature,
                "max_tokens": 8192}
                
                if top_p is not None:
                    model_kwargs["top_p"] = top_p
                if top_k is not None:
                    model_kwargs["top_k"] = top_k
                
                self.llm = ChatBedrock(
                    provider=MODEL_PROVIDER_3,
                    model_id=MODEL_ID_3,
                    region_name=AWS_REGION,
                    streaming=False,
                    model_kwargs=model_kwargs,
                    cache=False
                )
                self.llm.provider_stop_sequence_key_name_map = {'anthropic': 'stop_sequences', 'amazon': 'stopSequences',
                                                            'ai21': 'stop_sequences', 'cohere': 'stop_sequences',
                                                            'mistral': 'stop','meta': 'stop','custom': 'END_GENERATION'}
            else:

                # model_kwargs = {
                # "temperature": temperature,
                # "max_tokens": 15000}
                
                # if top_p is not None:
                #     model_kwargs["top_p"] = top_p
                # if top_k is not None:
                #     model_kwargs["top_k"] = top_k

                # self.llm = ChatBedrock(
                #     model=MODEL_ID_4,
                #     provider=MODEL_PROVIDER_4,
                #     region_name=AWS_REGION,
                #     streaming=False,
                #     model_kwargs=model_kwargs,
                #     cache=False
                # )
                # self.llm.provider_stop_sequence_key_name_map = {'anthropic': 'stop_sequences', 'amazon': 'stopSequences',
                #                                             'ai21': 'stop_sequences', 'cohere': 'stop_sequences',
                # 
                #                                             'mistral': 'stop','meta': 'stop','custom': 'END_GENERATION'}

                self.llm = LLM(
                                model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                                aws_access_key_id=AWS_ACCESS_KEY,
                                aws_secret_access_key=AWS_SECRET_KEY,
                                aws_region_name=AWS_REGION,
                                temperature=temperature,
                                max_tokens=8192)
                
                # self.llm.provider_stop_sequence_key_name_map = {'anthropic': 'stop_sequences', 'amazon': 'stopSequences',
                #                                             'ai21': 'stop_sequences', 'cohere': 'stop_sequences',
                #                                             'mistral': 'stop','meta': 'stop','custom': 'END_GENERATION','deepseek':'stop'}

            logging.info("---->Model Initialization Complete<-----")
            print("Model Initialization Complete: ", datetime.now())
            return self.llm

        except Exception as e:
            raise CustomException(e, sys)

