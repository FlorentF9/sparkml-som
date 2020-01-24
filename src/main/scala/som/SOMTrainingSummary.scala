package xyz.florentforest.spark.ml.som

import org.apache.spark.sql.DataFrame

class SOMTrainingSummary(val predictions: DataFrame,
                         val predictionCol: String,
                         val featuresCol: String,
                         val height: Int,
                         val width: Int,
                         val tMax: Double,
                         val tMin: Double,
                         val maxIter: Int,
                         val tol: Double,
                         val topology: String,
                         val neighborhoodKernel: String,
                         val temperatureDecay: String,
                         val trainingCost: Double,
                         val objectiveHistory: Array[Double]) extends Serializable
