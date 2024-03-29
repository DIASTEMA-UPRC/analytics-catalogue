import pymongo

from pyspark.sql import SparkSession
from pyspark.context import SparkContext

from logger import *

DEFAULT_HOST       = "0.0.0.0"
DEFAULT_PORT       = "9000"
DEFAULT_USER       = "minioadmin"
DEFAULT_PASS       = "minioadmin"
DEFAULT_MONGO_HOST = "0.0.0.0"
DEFAULT_MONGO_PORT = "27017"


# NOTE: Mongo support is an afterthought, and is not fully implemented. It is not recommended to use it in production.
class Storage:
    """
    This class represents a Storage connection config
    
    Attributes
    ----------
    host : str
        The address to connect to
    port : str
        The port to connect to
    username : str
        The username to connect with
    password : str
        The password to connect with
    mongo_host: str
        The address of the MongoDB server
    mongo_port: str
        The port of the MongoDB server
    """
    def __init__(self, host: str, port: str, username: str, password: str, mongo_host: str, mongo_port: str):
        self.host       = host
        self.port       = port
        self.username   = username
        self.password   = password
        self.minio      = MinIO(host, port, username, password)
        self.mongo_host = mongo_host
        self.mongo_port = mongo_port

    
    def __repr__(self) -> str:
        return f"Storage: {self.host}:{self.port}\n  Username: {self.username}\n  Password: {self.password}"


    def connect(self, ctx: SparkContext):
        """
        Set the Storage connection config for a given Spark context

        Parameters
        ----------
        ctx : SparkContext
            The Spark context to use
        """
        ctx._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        ctx._jsc.hadoopConfiguration().set("fs.s3a.connection.ssl.enabled", "false")
        ctx._jsc.hadoopConfiguration().set("fs.s3a.endpoint", f"http://{self.host}:{self.port}")
        ctx._jsc.hadoopConfiguration().set("fs.s3a.access.key", self.username)
        ctx._jsc.hadoopConfiguration().set("fs.s3a.secret.key", self.password)
        ctx._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")


    def connect_mongo(self) -> pymongo.MongoClient:
        """
        Connect to the MongoDB server

        Returns
        -------
        pymongo.MongoClient
            The MongoDB client
        """
        return pymongo.MongoClient(f"mongodb://{self.mongo_host}:{self.mongo_port}/")


    @staticmethod
    def get_from_runtime(session: SparkSession):
        """
        Generates a Storage object from the environment variables in the runtime configuration

        Parameters
        ----------
        session : SparkSession
            The Spark Session to get the runtime configuration from

        Returns
        -------
        Storage
            The resulting Storage object
        """
        host       = session.conf.get("spark.executorEnv.MINIO_HOST", DEFAULT_HOST)
        port       = session.conf.get("spark.executorEnv.MINIO_PORT", DEFAULT_PORT)
        username   = session.conf.get("spark.executorEnv.MINIO_USER", DEFAULT_USER)
        password   = session.conf.get("spark.executorEnv.MINIO_PASS", DEFAULT_PASS)
        mongo_host = session.conf.get("spark.executorEnv.MONGO_HOST", DEFAULT_MONGO_HOST)
        mongo_port = session.conf.get("spark.executorEnv.MONGO_PORT", DEFAULT_MONGO_PORT)

        LOGGER.setLevel(logging.WARN)

        if host == DEFAULT_HOST:
            LOGGER.warn(f"Using default value for MINIO_HOST: '{DEFAULT_HOST}'")
        if port == DEFAULT_PORT:
            LOGGER.warn(f"Using default value for MINIO_PORT: '{DEFAULT_PORT}'")
        if username == DEFAULT_USER:
            LOGGER.warn(f"Using default value for MINIO_USER: '{DEFAULT_USER}'")
        if password == DEFAULT_PASS:
            LOGGER.warn(f"Using default value for MINIO_PASS: '{DEFAULT_PASS}'")
        if mongo_host == DEFAULT_MONGO_HOST:
            LOGGER.warn(f"Using default value for MONGO_HOST: '{DEFAULT_MONGO_HOST}'")
        if mongo_port == DEFAULT_MONGO_PORT:
            LOGGER.warn(f"Using default value for MONGO_PORT: '{DEFAULT_MONGO_PORT}'")

        return Storage(host, port, username, password, mongo_host, mongo_port)


from minio import Minio

class MinIO(Minio):
    def __init__(self, host, port, username, password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.endpoint = f"{host}:{port}"

        super().__init__(self.endpoint, access_key=self.username, secret_key=self.password, secure=False)

        try:
            if not self.bucket_exists("diastemaviz"):
                self.make_bucket("diastemaviz")
        except:
            raise Exception("Failed to connect to MinIO")

    @staticmethod
    def get_from_runtime(session: SparkSession):
        host       = session.conf.get("spark.executorEnv.MINIO_HOST", DEFAULT_HOST)
        port       = session.conf.get("spark.executorEnv.MINIO_PORT", DEFAULT_PORT)
        username   = session.conf.get("spark.executorEnv.MINIO_USER", DEFAULT_USER)
        password   = session.conf.get("spark.executorEnv.MINIO_PASS", DEFAULT_PASS)

        return Storage(host, port, username, password)
