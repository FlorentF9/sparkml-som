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

import org.apache.spark.ml.param._
import org.apache.spark.sql.types.{IntegerType, StructType}
import xyz.florentforest.spark.ml.util.SchemaUtils

trait SOMParams extends Params with HasMaxIter with HasFeaturesCol with HasSeed with HasPredictionCol with HasTol {

  /**
    * The height of the map to create (height). Must be &gt; 1.
    * Default: 10.
    */
  final val height = new IntParam(this, "height", "The height of the map to create. " +
    "Must be > 1.", ParamValidators.gt(1))

  def getHeight: Int = $(height)

  /**
    * The width of the map to create (width). Must be &gt; 1.
    * Default: 10.
    */
  final val width = new IntParam(this, "width", "The width of the map to create. " +
    "Must be > 1.", ParamValidators.gt(1))

  def getWidth: Int = $(width)

  /**
    * Initial temperature parameter value (tMax). Must be &gt; 0.0.
    * Default: 10.0.
    */
  final val tMax = new DoubleParam(this, "tMax", "The initial temperature parameter. " +
    "Must be > 0.0.", ParamValidators.gt(0.0))

  def getTMax: Double = $(tMax)

  /**
    * Final temperature parameter value (tMin). Must be &gt; 0.0.
    * Default: 0.1.
    */
  final val tMin = new DoubleParam(this, "tMin", "The final temperature parameter. " +
    "Must be > 0.0.", ParamValidators.gt(0.0))

  def getTMin: Double = $(tMin)

  /**
    * Param for the map grid topology type. Only "rectangular" is available at the moment, hexagonal will soon be added.
    * Default: rectangular.
    */
  final val topology = new Param[String](this, "topology", "The map grid topology type. " +
    "Supported options: 'rectangular'.",
    ParamValidators.inArray(Array("rectangular")))

  def getTopology: String = $(topology)

  /**
    * Param for the neighborhood kernel type. This can be either "gaussian" or "rectangular". Default: gaussian.
    */
  final val neighborhoodKernel = new Param[String](this, "neighborhoodKernel", "The neighborhood kernel type. " +
    "Supported options: 'gaussian' and 'rectangular'.",
    ParamValidators.inArray(Array("gaussian", "rectangular")))

  def getNeighborhoodKernel: String = $(neighborhoodKernel)

  /**
    * Param for the temperature decay type. This can be either "exponential" or "linear". Default: exponential.
    */
  final val temperatureDecay = new Param[String](this, "temperatureDecay", "The temperature decay type. " +
    "Supported options: 'exponential' and 'linear'.",
    ParamValidators.inArray(Array("exponential", "linear")))

  def getTemperatureDecay: String = $(temperatureDecay)

  /**
    * Validates and transforms the input schema.
    * @param schema input schema
    * @return output schema
    */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.appendColumn(schema, $(predictionCol), IntegerType)
  }

}

/**
  * Trait for shared param maxIter.
  */
 trait HasMaxIter extends Params {

  /**
    * Param for maximum number of iterations (&gt;= 0).
    */
  final val maxIter: IntParam = new IntParam(this, "maxIter", "maximum number of iterations (>= 0)", ParamValidators.gtEq(0))

  final def getMaxIter: Int = $(maxIter)
}

/**
  * Trait for shared param featuresCol (default: "features").
  */
 trait HasFeaturesCol extends Params {

  /**
    * Param for features column name.
    */
  final val featuresCol: Param[String] = new Param[String](this, "featuresCol", "features column name")

  setDefault(featuresCol, "features")

  final def getFeaturesCol: String = $(featuresCol)
}

/**
  * Trait for shared param predictionCol (default: "prediction").
  */
 trait HasPredictionCol extends Params {

  /**
    * Param for prediction column name.
    */
  final val predictionCol: Param[String] = new Param[String](this, "predictionCol", "prediction column name")

  setDefault(predictionCol, "prediction")

  final def getPredictionCol: String = $(predictionCol)
}

/**
  * Trait for shared param seed (default: this.getClass.getName.hashCode.toLong).
  */
 trait HasSeed extends Params {

  /**
    * Param for random seed.
    */
  final val seed: LongParam = new LongParam(this, "seed", "random seed")

  setDefault(seed, this.getClass.getName.hashCode.toLong)

  final def getSeed: Long = $(seed)
}

/**
  * Trait for shared param tol.
  */
 trait HasTol extends Params {

  /**
    * Param for the convergence tolerance for iterative algorithms (&gt;= 0).
    */
  final val tol: DoubleParam = new DoubleParam(this, "tol", "the convergence tolerance for iterative algorithms (>= 0)", ParamValidators.gtEq(0))

  final def getTol: Double = $(tol)
}