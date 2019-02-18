package xyz.florentforest.spark.ml.som

import org.apache.spark.ml.Model
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Row}

class SOMModel(override val uid: String, val prototypes: Array[Vector]) extends Model[SOMModel] with SOMParams {

  private val prototypesWithNorm =
    if (prototypes == null) null else prototypes.map(new VectorWithNorm(_))

  override def copy(extra: ParamMap): SOMModel = {
    copyValues(new SOMModel(uid, prototypes), extra)
  }

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)

    val predictUDF = udf((vector: Vector) => predict(vector))

    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  def predict(features: Vector): Int = {
    SOM.findClosest(prototypesWithNorm, new VectorWithNorm(features))._1
  }

  def computeCost(dataset: Dataset[_]): Double = {
    val bcPrototypesWithNorm = dataset.sparkSession.sparkContext.broadcast(prototypesWithNorm)
    dataset.select(col($(featuresCol))).rdd.map {
      case Row(point: Vector) => SOM.pointCost(bcPrototypesWithNorm.value, new VectorWithNorm(point))
    }.sum()
  }

}