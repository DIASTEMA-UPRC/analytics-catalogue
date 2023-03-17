import logging

from Job import Job
from logger import LOGGER


def get_classification_model_and_params(job: Job):
    params_db = job.storage.connect_mongo()["Diastema"]["datatoolkit"]
    params = params_db.find_one({ "job-id": job.args.job_id })

    if not params:
        params = dict()

        LOGGER.setLevel(logging.DEBUG)
        LOGGER.fatal(f"No params found, using default values for job_id: '{job.args.job_id}'")

    model = None
    algorithm = job.args.algorithm.upper()

    if algorithm == "LOGISTICREGRESSION":
        from pyspark.ml.classification import LogisticRegression

        maxIter = params.get("maxIter", 100)
        regParam = params.get("regParam", 0.0)
        elasticNetParam = params.get("elasticNetParam", 0.0)
        tol = float(params.get("tol", 1e-6))
        fitIntercept = params.get("fitIntercept", True)
        threshold = params.get("threshold", 0.5)
        standardization = params.get("standardization", True)
        aggregationDepth = params.get("aggregationDepth", 2)
        family = params.get("family", "auto")
        maxBlockSizeInMB = params.get("maxBlockSizeInMB", 256)

        model = LogisticRegression(labelCol=job.args.target_column, featuresCol="features", maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam, tol=tol, fitIntercept=fitIntercept, threshold=threshold, standardization=standardization, aggregationDepth=aggregationDepth, family=family, maxBlockSizeInMB=maxBlockSizeInMB)
    elif algorithm == "DECISIONTREE":
        from pyspark.ml.classification import DecisionTreeClassifier

        maxDepth = params.get("maxDepth", 5)
        maxBins = params.get("maxBins", 32)
        minInstancesPerNode = params.get("minInstancesPerNode", 1)
        minInfoGain = params.get("minInfoGain", 0.0)
        maxMemoryInMB = params.get("maxMemoryInMB", 256)
        cacheNodeIds = params.get("cacheNodeIds", False)
        checkpointInterval = params.get("checkpointInterval", 10)
        impurity = params.get("impurity", "gini")
        minWeightFractionPerNode = params.get("minWeightFractionPerNode", 0.0)

        model = DecisionTreeClassifier(labelCol=job.args.target_column, featuresCol="features", maxDepth=maxDepth, maxBins=maxBins, minInstancesPerNode=minInstancesPerNode, minInfoGain=minInfoGain, maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds, checkpointInterval=checkpointInterval, impurity=impurity, minWeightFractionPerNode=minWeightFractionPerNode)
    elif algorithm == "RANDOMFOREST":
        from pyspark.ml.classification import RandomForestClassifier

        maxDepth = params.get("maxDepth", 5)
        maxBins = params.get("maxBins", 32)
        minInstancesPerNode = params.get("minInstancesPerNode", 1)
        minInfoGain = params.get("minInfoGain", 0.0)
        maxMemoryInMB = params.get("maxMemoryInMB", 256)
        cacheNodeIds = params.get("cacheNodeIds", False)
        checkpointInterval = params.get("checkpointInterval", 10)
        impurity = params.get("impurity", "gini")
        numTrees = params.get("numTrees", 20)
        subsamplingRate = params.get("subsamplingRate", 1.0)
        minWeightFractionPerNode = params.get("minWeightFractionPerNode", 0.0)

        model = RandomForestClassifier(labelCol=job.args.target_column, featuresCol="features", maxDepth=maxDepth, maxBins=maxBins, minInstancesPerNode=minInstancesPerNode, minInfoGain=minInfoGain, maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds, checkpointInterval=checkpointInterval, impurity=impurity, numTrees=numTrees, subsamplingRate=subsamplingRate, minWeightFractionPerNode=minWeightFractionPerNode)
    elif algorithm == "GBT":
        from pyspark.ml.classification import GBTClassifier

        maxDepth = params.get("maxDepth", 5)
        maxBins = params.get("maxBins", 32)
        minInstancesPerNode = params.get("minInstancesPerNode", 1)
        minInfoGain = params.get("minInfoGain", 0.0)
        maxMemoryInMB = params.get("maxMemoryInMB", 256)
        cacheNodeIds = params.get("cacheNodeIds", False)
        checkpointInterval = params.get("checkpointInterval", 10)
        maxIter = params.get("maxIter", 20)
        stepSize = params.get("stepSize", 0.1)
        subsamplingRate = params.get("subsamplingRate", 1.0)
        validationTol = float(params.get("validationTol", 0.01))
        minWeightFractionPerNode = params.get("minWeightFractionPerNode", 0.0)

        model = GBTClassifier(labelCol=job.args.target_column, featuresCol="features", maxDepth=maxDepth, maxBins=maxBins, minInstancesPerNode=minInstancesPerNode, minInfoGain=minInfoGain, maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds, checkpointInterval=checkpointInterval, maxIter=maxIter, stepSize=stepSize, subsamplingRate=subsamplingRate, validationTol=validationTol, minWeightFractionPerNode=minWeightFractionPerNode)
    elif algorithm == "MLP":
        from pyspark.ml.classification import MultilayerPerceptronClassifier
    
        maxIter = params.get("maxIter", 100)
        tol = float(params.get("tol", 1e-6))
        blockSize = params.get("blockSize", 128)
        stepSize = params.get("stepSize", 0.03)
        solver = params.get("solver", "l-bfgs")

        model = MultilayerPerceptronClassifier(labelCol=job.args.target_column, featuresCol="features", maxIter=maxIter, tol=tol, blockSize=blockSize, stepSize=stepSize, solver=solver)
    elif algorithm == "LINEARSVC":
        from pyspark.ml.classification import LinearSVC

        maxIter = params.get("maxIter", 100)
        regParam = params.get("regParam", 0.0)
        tol = float(params.get("tol", 1e-6))
        fitIntercept = params.get("fitIntercept", True)
        standardization = params.get("standardization", True)
        threshold = params.get("threshold", 0.0)
        aggregationDepth = params.get("aggregationDepth", 2)
        maxBlockSizeInMB = params.get("maxBlockSizeInMB", 256)

        model = LinearSVC(labelCol=job.args.target_column, featuresCol="features", maxIter=maxIter, regParam=regParam, tol=tol, fitIntercept=fitIntercept, standardization=standardization, threshold=threshold, aggregationDepth=aggregationDepth, maxBlockSizeInMB=maxBlockSizeInMB)
    elif algorithm == "ONEVSREST":
        from pyspark.ml.classification import OneVsRest

        model = OneVsRest(labelCol=job.args.target_column, featuresCol="features")
    elif algorithm == "NAIVEBAYES":
        from pyspark.ml.classification import NaiveBayes

        smoothing = params.get("smoothing", 1.0)
        modelType = params.get("modelType", "multinomial")

        model = NaiveBayes(labelCol=job.args.target_column, featuresCol="features", smoothing=smoothing, modelType=modelType)
    elif algorithm == "FM":
        from pyspark.ml.classification import FMClassifier

        numFactors = params.get("numFactors", 8)
        fitIntercept = params.get("fitIntercept", True)
        fitLinear = params.get("fitLinear", True)
        regParam = params.get("regParam", 0.0)
        miniBatchFraction = params.get("miniBatchFraction", 1.0)
        initStd = params.get("initStd", 0.01)
        maxIter = params.get("maxIter", 100)
        stepSize = params.get("stepSize", 1.0)
        tol = float(params.get("tol", 1e-6))
        solver = params.get("solver", "adamW")

        model = FMClassifier(labelCol=job.args.target_column, featuresCol="features", numFactors=numFactors, fitIntercept=fitIntercept, fitLinear=fitLinear, regParam=regParam, miniBatchFraction=miniBatchFraction, initStd=initStd, maxIter=maxIter, stepSize=stepSize, tol=tol, solver=solver)
    else:
        LOGGER.setLevel(logging.FATAL)
        LOGGER.fatal(f"Unknown classification algorithm: '{job.args.algorithm}'")

    return model


def get_regression_model_and_params(job: Job):
    params_db = job.storage.connect_mongo()["Diastema"]["datatoolkit"]
    params = params_db.find_one({ "job-id": job.args.job_id })

    if not params:
        params = dict()

        LOGGER.setLevel(logging.DEBUG)
        LOGGER.fatal(f"No params found, using default values for job_id: '{job.args.job_id}'")

    model = None
    algorithm = job.args.algorithm.upper()

    if algorithm == "LINEARREGRESSION":
        from pyspark.ml.regression import LinearRegression

        maxIter = params.get("maxIter", 100)
        regParam = params.get("regParam", 0.0)
        elasticNetParam = params.get("elasticNetParam", 0.0)
        tol = float(params.get("tol", 1e-6))
        fitIntercept = params.get("fitIntercept", True)
        standardization = params.get("standardization", True)
        solver = params.get("solver", "auto")
        aggregationDepth = params.get("aggregationDepth", 2)
        loss = params.get("loss", "squaredError")
        epsilon = params.get("epsilon", 1.35)
        maxBlockSizeInMB = params.get("maxBlockSizeInMB", 0.0)

        model = LinearRegression(labelCol=job.args.target_column, featuresCol="features", maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam, tol=tol, fitIntercept=fitIntercept, standardization=standardization, solver=solver, aggregationDepth=aggregationDepth, loss=loss, epsilon=epsilon, maxBlockSizeInMB=maxBlockSizeInMB)
    elif algorithm == "DECISIONTREE":
        from pyspark.ml.regression import DecisionTreeRegressor

        maxDepth = params.get("maxDepth", 5)
        maxBins = params.get("maxBins", 32)
        minInstancesPerNode = params.get("minInstancesPerNode", 1)
        minInfoGain = params.get("minInfoGain", 0.0)
        maxMemoryInMB = params.get("maxMemoryInMB", 256)
        cacheNodeIds = params.get("cacheNodeIds", False)
        checkpointInterval = params.get("checkpointInterval", 10)
        minWeightFractionPerNode = params.get("minWeightFractionPerNode", 0.0)

        model = DecisionTreeRegressor(labelCol=job.args.target_column, featuresCol="features", maxDepth=maxDepth, maxBins=maxBins, minInstancesPerNode=minInstancesPerNode, minInfoGain=minInfoGain, maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds, checkpointInterval=checkpointInterval, minWeightFractionPerNode=minWeightFractionPerNode)
    elif algorithm == "RANDOMFOREST":
        from pyspark.ml.regression import RandomForestRegressor

        maxDepth = params.get("maxDepth", 5)
        maxBins = params.get("maxBins", 32)
        minInstancesPerNode = params.get("minInstancesPerNode", 1)
        minInfoGain = params.get("minInfoGain", 0.0)
        maxMemoryInMB = params.get("maxMemoryInMB", 256)
        cacheNodeIds = params.get("cacheNodeIds", False)
        checkpointInterval = params.get("checkpointInterval", 10)
        subsamplingRate = params.get("subsamplingRate", 1.0)
        numTrees = params.get("numTrees", 20)
        minWeightFractionPerNode = params.get("minWeightFractionPerNode", 0.0)

        model = RandomForestRegressor(labelCol=job.args.target_column, featuresCol="features", maxDepth=maxDepth, maxBins=maxBins, minInstancesPerNode=minInstancesPerNode, minInfoGain=minInfoGain, maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds, checkpointInterval=checkpointInterval, subsamplingRate=subsamplingRate, numTrees=numTrees, minWeightFractionPerNode=minWeightFractionPerNode)
    elif algorithm == "GBT":
        from pyspark.ml.regression import GBTRegressor

        maxDepth = params.get("maxDepth", 5)
        maxBins = params.get("maxBins", 32)
        minInstancesPerNode = params.get("minInstancesPerNode", 1)
        minInfoGain = params.get("minInfoGain", 0.0)
        maxMemoryInMB = params.get("maxMemoryInMB", 256)
        cacheNodeIds = params.get("cacheNodeIds", False)
        subsamplingRate = params.get("subsamplingRate", 1.0)
        checkpointInterval = params.get("checkpointInterval", 10)
        lossType = params.get("lossType", "squared")
        maxIter = params.get("maxIter", 20)
        stepSize = params.get("stepSize", 0.1)
        minWeightFractionPerNode = params.get("minWeightFractionPerNode", 0.0)

        model = GBTRegressor(labelCol=job.args.target_column, featuresCol="features", maxDepth=maxDepth, maxBins=maxBins, minInstancesPerNode=minInstancesPerNode, minInfoGain=minInfoGain, maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds, subsamplingRate=subsamplingRate, checkpointInterval=checkpointInterval, lossType=lossType, maxIter=maxIter, stepSize=stepSize, minWeightFractionPerNode=minWeightFractionPerNode)
    elif algorithm == "SURVIVAL":
        from pyspark.ml.regression import AFTSurvivalRegression

        fitIntercept = params.get("fitIntercept", True)
        maxIter = params.get("maxIter", 100)
        tol = float(params.get("tol", 1e-6))
        aggregationDepth = params.get("aggregationDepth", 2)
        maxBlockSizeInMB = params.get("maxBlockSizeInMB", 0)

        model = AFTSurvivalRegression(labelCol=job.args.target_column, featuresCol="features", fitIntercept=fitIntercept, maxIter=maxIter, tol=tol, aggregationDepth=aggregationDepth, maxBlockSizeInMB=maxBlockSizeInMB)
    elif algorithm == "ISOTONIC":
        from pyspark.ml.regression import IsotonicRegression

        isotonic = params.get("isotonic", True)

        model = IsotonicRegression(labelCol=job.args.target_column, featuresCol="features", isotonic=isotonic)
    elif algorithm == "FM":
        from pyspark.ml.regression import FMRegressor

        numFactors = params.get("numFactors", 8)
        fitIntercept = params.get("fitIntercept", True)
        fitLinear = params.get("fitLinear", True)
        regParam = params.get("regParam", 0.0)
        miniBatchFraction = params.get("miniBatchFraction", 1.0)
        initStd = params.get("initStd", 0.01)
        maxIter = params.get("maxIter", 100)
        stepSize = params.get("stepSize", 1.0)
        tol = float(params.get("tol", 1e-6))
        solver = params.get("solver", "adamW")

        model = FMRegressor(labelCol=job.args.target_column, featuresCol="features", numFactors=numFactors, fitIntercept=fitIntercept, fitLinear=fitLinear, regParam=regParam, miniBatchFraction=miniBatchFraction, initStd=initStd, maxIter=maxIter, stepSize=stepSize, tol=tol, solver=solver)
    else:
        LOGGER.setLevel(logging.FATAL)
        LOGGER.fatal(f"Unknown regression algorithm: '{job.args.algorithm}'")

    return model


def get_clustering_model_and_params(job: Job):
    params_db = job.storage.connect_mongo()["Diastema"]["datatoolkit"]
    params = params_db.find_one({ "job-id": job.args.job_id })

    if not params:
        params = dict()

        LOGGER.setLevel(logging.DEBUG)
        LOGGER.fatal(f"No params found, using default values for job_id: '{job.args.job_id}'")

    model = None
    algorithm = job.args.algorithm.upper()

    if algorithm == "KMEANS":
        from pyspark.ml.clustering import KMeans

        k = params.get("k", 2)
        initMode = params.get("initMode", "k-means||")
        initSteps = params.get("initSteps", 5)
        tol = float(params.get("tol", 0.0001))
        maxIter = params.get("maxIter", 20)
        distanceMeasure = params.get("distanceMeasure", "euclidean")

        model = KMeans(k=k, initMode=initMode, initSteps=initSteps, tol=tol, maxIter=maxIter, distanceMeasure=distanceMeasure)
    elif algorithm == "LDA":
        from pyspark.ml.clustering import LDA

        maxIter = params.get("maxIter", 20)
        checkpointInterval = params.get("checkpointInterval", 10)
        k = params.get("k", 10)
        optimizer = params.get("optimizer", "online")
        learningOffset = params.get("learningOffset", 1024.0)
        learningDecay = params.get("learningDecay", 0.51)
        subsamplingRate = params.get("subsamplingRate", 0.05)
        optimizeDocConcentration = params.get("optimizeDocConcentration", True)
        keepLastCheckpoint = params.get("keepLastCheckpoint", True)

        model = LDA(maxIter=maxIter, checkpointInterval=checkpointInterval, k=k, optimizer=optimizer, learningOffset=learningOffset, learningDecay=learningDecay, subsamplingRate=subsamplingRate, optimizeDocConcentration=optimizeDocConcentration, keepLastCheckpoint=keepLastCheckpoint)
    elif algorithm == "GMM":
        from pyspark.ml.clustering import GaussianMixture

        k = params.get("k", 2)
        tol = float(params.get("tol", 0.01))
        maxIter = params.get("maxIter", 100)
        aggregationDepth = params.get("aggregationDepth", 2)

        model = GaussianMixture(k=k, tol=tol, maxIter=maxIter, aggregationDepth=aggregationDepth)
    elif algorithm == "PIC":
        from pyspark.ml.clustering import PowerIterationClustering
        
        k = params.get("k", 2)
        maxIter = params.get("maxIter", 20)
        initMode = params.get("initMode", "random")

        model = PowerIterationClustering(k=k, maxIter=maxIter, initMode=initMode)
    else:
        LOGGER.setLevel(logging.FATAL)
        LOGGER.fatal(f"Unknown clustering algorithm: '{job.args.algorithm}'")

    return model
