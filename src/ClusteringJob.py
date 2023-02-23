import pandas as pd

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

from Job import Job
from logger import *

import time
import psutil
import sys


def main():
    job = Job()

    if not job.args:
        return 1

    tic = time.perf_counter()

    # Find algorithm
    if job.args.algorithm.upper() == "KMEANS":
        model = KMeans()
    else:
        LOGGER.setLevel(logging.FATAL)
        LOGGER.fatal(f"Unknown classification algorithm: '{job.args.algorithm}'")
        return 1
    
    LOGGER.setLevel(logging.DEBUG)
    
    # Load data
    LOGGER.debug("Loading data...")
    raw = job.spark.read.option("header", True).option("inferSchema", True).csv(f"s3a://{job.args.input_path}/*")
    LOGGER.debug("Loaded data")

    # Assemble data
    LOGGER.debug("Assembling data...")
    assembler = VectorAssembler(inputCols=raw.columns, outputCol="features")
    data = assembler.transform(raw)
    LOGGER.debug("Assembled data")

    # Fit and transform
    LOGGER.debug("Fitting...")
    model = model.fit(data)
    LOGGER.debug("Fitted")

    # Export results
    LOGGER.debug("Exporting results...")
    out = model.clusterCenters()[0]
    out = pd.DataFrame(out, columns=["clusterCenters"])
    out = job.spark.createDataFrame(out)
    out.repartition(1).write.csv(path=f"s3a://{job.args.output_path}", header="true", mode="overwrite")
    
    toc = time.perf_counter()
    execution_speed = (toc - tic) * 1000
    ram_usage = (psutil.virtual_memory().total - psutil.virtual_memory().free) / 1024 / 1024
    ram_existing = psutil.virtual_memory().total / 1024 / 1024
    disk_usage = sys.getsizeof(model) / 1024 / 1024

    performance = {
        "ram-usage": int(ram_usage),
        "ram-existing": int(ram_existing),
        "disk-usage": int(disk_usage),
        "execution-speed": int(execution_speed)
    }

    performance_db = job.storage.connect_mongo()["UIDB"]["pipelines"]
    performance_db.update_one({"analysisid": job.args.analysis_id}, {"$set": {f"performance.{job.args.job_id}": performance}}, upsert=True)
    
    LOGGER.debug("Exported results")

    return 0

if __name__ == "__main__":
    code = main()
    exit(code)
