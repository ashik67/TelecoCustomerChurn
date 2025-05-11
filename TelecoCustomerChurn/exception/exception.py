import sys
from TelecoCustomerChurn.logging import logger
class CustomerChurnException(Exception):
    """
    Custom exception class for the TelecoCustomerChurn project.
    Provides detailed error messages including file name and line number.
    """
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = CustomerChurnException.get_detailed_error_message(
            error_message, error_detail
        )

    @staticmethod
    def get_detailed_error_message(error_message, error_detail):
        """
        Returns a detailed error message with file name and line number.
        """
        import sys
        import traceback

        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error in script: [{file_name}] at line [{line_number}]: {error_message}"
        else:
            return f"Error: {error_message}"

    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    try:
        logger.logging.info("Logging is working")
        raise ValueError("some error")
    except ValueError as e:
        raise CustomerChurnException("some errr", sys) from e