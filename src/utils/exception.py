"""Custom exception handling with detailed error logging."""

import sys
from src.utils.logger import setup_logger

logger = setup_logger()


def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error message including file name and line number.

    Args:
        error: Exception object
        error_detail: sys module to get traceback details

    Returns:
        str: Formatted error message
    """
    _, _, exc_tb = error_detail.exc_info()
    error_file_name = exc_tb.tb_frame.f_code.co_filename
    error_line_number = exc_tb.tb_lineno
    error_message_str = str(error)

    error_message = (
        f"Error occurred in script: [{error_file_name}] "
        f"at line: [{error_line_number}] "
        f"with message: [{error_message_str}]"
    )
    return error_message


class CustomException(Exception):
    """Custom Exception class for application-wide error handling."""

    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException with detailed error information.

        Args:
            error_message: Exception message
            error_detail: sys module for traceback extraction
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
