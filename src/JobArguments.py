import sys

from logger import *


class JobArguments:
    """
    This class represents the system arguments that are passed during a job creation
    
    Attributes
    ----------
    algorithm : str
        The type of algorithm to use
    input_path : str
        The path to for the input data
    output_path : str
        The path to export the resulting data of the job
    target_column : str
        The column to target with the algorithm
    job_id : str
        The id of the job
    """
    def __init__(self, algorithm: str, input_path: str, output_path: str, target_column: str, job_id: str, analysis_id: str):
        self.algorithm     = algorithm
        self.input_path    = input_path
        self.output_path   = output_path
        self.target_column = target_column
        self.job_id        = job_id
        self.analysis_id   = analysis_id


    def __repr__(self) -> str:
        return f"{self.algorithm.upper()} Job\n  Input Path: {self.input_path}\n  Output Path: {self.output_path}\n  Target Column: {self.target_column}"


    @staticmethod
    def get_from_runtime():
        """
        Generates a JobArguments object from the system arguments that are passed job creation

        Returns
        -------
        JobArguments
            The resulting JobArguments object
        """
        if len(sys.argv) < 6:
            LOGGER.setLevel(logging.WARN)
            LOGGER.warn("Missing job arguments from runtime!")

            return None

        algorithm = sys.argv[1]
        input_path = sys.argv[2]
        output_path = sys.argv[3]
        target_column = sys.argv[4]
        job_id = sys.argv[5]
        analysis_id = sys.argv[6]

        return JobArguments(algorithm, input_path, output_path, target_column, job_id, analysis_id)
