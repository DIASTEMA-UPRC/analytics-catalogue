from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.functions import col

from Job import Job
from logger import *

import time
import psutil
import sys

from visualization import visualization_df


def main():
    job = Job()

    if not job.args:
        return 1

    tic = time.perf_counter()

    # Find algorithm
    if job.args.algorithm.upper() == "DECISIONTREE":
        model = DecisionTreeClassifier(labelCol=job.args.target_column, featuresCol="features")
    elif job.args.algorithm.upper() == "RANDOMFOREST":
        model = RandomForestClassifier(labelCol=job.args.target_column, featuresCol="features")
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
    features = raw.drop(job.args.target_column).columns
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    data = assembler.transform(raw)
    train, test = data.randomSplit([0.8, 0.2])
    LOGGER.debug("Assembled data")

    # Fit and transform
    LOGGER.debug("Fitting and transforming...")
    model = model.fit(train)
    pred = model.transform(test)
    LOGGER.debug("Fitted and transformed")

    metricPred = pred.withColumn(job.args.target_column, col(job.args.target_column).cast("double"))
    predictionAndLabels = metricPred.select(job.args.target_column, "prediction").rdd
    metrics = MulticlassMetrics(predictionAndLabels)
    f1 = metrics.weightedFMeasure()
    acc = metrics.accuracy
    confusion = metrics.confusionMatrix().toArray().tolist()

    # Export results
    LOGGER.debug("Exporting results...")

    out = pred.select(job.args.target_column, "prediction")
    out.repartition(1).write.csv(path=f"s3a://{job.args.output_path}", header="true", mode="overwrite")
    visualization_df(job.storage.minio, out.toPandas(), job.args.job_id)

    db = job.storage.connect_mongo()["Diastema"]["Analytics"]
    db.insert_one({ "job_id": job.args.job_id, "f1": f1, "accuracy": acc, "confusion": confusion })

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
