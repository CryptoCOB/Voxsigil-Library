"""
Mock wandb module for projects that don't need wandb functionality
but have dependencies that import it
"""


class Proto:
    """Mock proto module"""

    class wandb_internal_pb2:
        """Mock wandb_internal_pb2 module"""

        class ErrorInfo:
            """Mock ErrorInfo class"""

            UNKNOWN = 0
            COMMUNICATION = 1
            AUTHENTICATION = 2
            USAGE = 3
            UNSUPPORTED = 4
