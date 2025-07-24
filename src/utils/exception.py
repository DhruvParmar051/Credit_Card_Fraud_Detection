import sys
from src.utils.logger import setup_logger

logger = setup_logger()

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    error_file_name = exc_tb.tb_frame.f_code.co_filename
    error_line_number = exc_tb.tb_lineno
    error_message_str = str(error)
    
    error_message = f"Error has occured in python script name [{error_file_name}] line number [{error_line_number}] error message [{error_message_str}]"
    
    return error_message

    

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        
    def __str__(self):
        return self.error_message
    
