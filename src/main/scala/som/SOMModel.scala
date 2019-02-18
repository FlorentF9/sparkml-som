/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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