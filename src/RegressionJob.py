from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col
from pyspark.mllib.evaluation import RegressionMetrics

from Job import Job
from logger import *
from params import get_regression_model_and_params

import time
import psutil
import sys

from visualization import visualization_df


def main():
    job = Job()

    if not job.args:
        return 1

    tic = time.perf_counter()

    model = get_regression_model_and_params(job)

    if not model:
        return 1

    LOGGER.setLevel(logging.DEBUG)

    # Load data
    LOGGER.debug("Loading data...")
    raw = job.spark.read.option("header", True).option("inferSchema", True).csv(f"s3a://{job.args.input_path}/*")
    LOGGER.debug("Loaded data")

    # Assemble data
    LOGGER.debug("Assembling data...")
    features = raw.drop(job.args.target_column).columns
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    train, test = raw.randomSplit([0.8, 0.2])
    LOGGER.debug("Assembled data")

    # Fit and transform
    LOGGER.debug("Fitting and transforming...")
    pipeline = Pipeline(stages=[assembler, model])
    pipelineModel = pipeline.fit(train)
    pred = pipelineModel.transform(test)
    LOGGER.debug("Fitted and transformed")

    LOGGER.debug("Saving model...")
    pipelineModel.write().overwrite(f"s3a://diastemamodels/{job.args.job_id}")

    metricPred = pred.withColumn(job.args.target_column, col(job.args.target_column).cast("double"))
    predictionAndLabels = metricPred.select(job.args.target_column, "prediction").rdd
    metrics = RegressionMetrics(predictionAndLabels)
    r2 = metrics.r2
    rmse = metrics.rootMeanSquaredError

    # Export results
    LOGGER.debug("Exporting results...")
    
    out = pred.select(job.args.target_column, "prediction")
    out.repartition(1).write.csv(path=f"s3a://{job.args.output_path}", header="true", mode="overwrite")
    visualization_df(job.storage.minio, out.toPandas(), job.args.job_id)

    db = job.storage.connect_mongo()["Diastema"]["Analytics"]
    db.insert_one({ "job_id": job.args.job_id, "r2": r2, "rmse": rmse })
    
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

    LOGGER.debug("Adding to performance database on analysis id: " + job.args.analysis_id)

    performance_db = job.storage.connect_mongo()["UIDB"]["pipelines"]
    performance_db.update_one({"analysisid": job.args.analysis_id}, {"$set": {f"performance.{job.args.job_id}": performance}}, upsert=True)
    
    LOGGER.debug("Exported results")

    return 0

if __name__ == "__main__":
    code = main()
    exit(code)
