import sys
from src.logger import logging

def error_msg_details(error,error_details:sys):
    _,_,exc_tb = error_details.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_num = exc_tb.tb_lineno
    err = str(error)

    error_message = f'The error is python script name {file_name}, in line number {line_num} and error message as {err}'

    return error_message


class CustomException(Exception):
    def __init__(self, error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_msg_details(error=error_message,error_details=error_details)

    def __str__(self):
        return self.error_message