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

import java.nio.ByteBuffer
import java.util.{Random => JavaRandom}

import breeze.numerics.{abs, exp, pow}
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import xyz.florentforest.spark.ml.util.MLUtils

import scala.util.hashing.MurmurHash3
import xyz.florentforest.spark.ml.linalg.BLAS.{axpy, scal}

class SOM(override val uid: String) extends Estimator[SOMModel] with SOMParams {

  setDefault(
    height -> 10,
    width -> 10,
    tMax -> 10.0,
    tMin -> 1.0,
    maxIter -> 20,
    tol -> 1e-4,
    topology -> "rectangular",
    neighborhoodKernel -> "gaussian",
    temperatureDecay -> "exponential")

  override def copy(extra: ParamMap): SOM = defaultCopy(extra)

  def this() = this(Identifiable.randomUID("SOM"))

  def setFeaturesCol(value: String): this.type = set(featuresCol, value)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setHeight(value: Int): this.type = set(height, value)

  def setWidth(value: Int): this.type = set(width, value)

  def setTMax(value: Double): this.type = set(tMax, value)

  def setTMin(value: Double): this.type = set(tMin, value)

  def setTopology(value: String): this.type = set(topology, value)

  def setNeighborhoodKernel(value: String): this.type = set(neighborhoodKernel, value)

  def setTemperatureDecay(value: String): this.type = set(temperatureDecay, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setTol(value: Double): this.type = set(tol, value)

  def setSeed(value: Long): this.type = set(seed, value)

  override def fit(dataset: Dataset[_]): SOMModel = {
    transformSchema(dataset.schema, logging = true)

    val handlePersistence = dataset.storageLevel == StorageLevel.NONE

    val instances: RDD[Vector] = dataset.select(col($(featuresCol))).rdd.map {
      case Row(point: Vector) => point
    }

    if (handlePersistence) {
      instances.persist(StorageLevel.MEMORY_AND_DISK)
    }

    val parentModel = run(instances)
    val model = copyValues(new SOMModel(uid, parentModel.prototypes).setParent(this))

    if (handlePersistence) {
      instances.unpersist()
    }
    model

  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  // Initial prototypes can be provided as a SOMModel object rather than using the
  // random initialization.
  private var initialModel: Option[SOMModel] = None

  def setInitialModel(model: SOMModel): this.type = {
    require(model.height == height, "mismatched map height")
    require(model.width == width, "mismatched map width")
    initialModel = Some(model)
    this
  }

  def run(data: RDD[Vector]): SOMModel = {

    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    // Compute squared norms and cache them.
    val norms = data.map(Vectors.norm(_, 2.0))
    norms.persist()
    val zippedData = data.zip(norms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }
    val model = runAlgorithm(zippedData)
    norms.unpersist()

    // Warn at the end of the run as well, for increased visibility.
    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data was not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    model
  }

  def runAlgorithm(data: RDD[VectorWithNorm]): SOMModel = {

    val sc = data.sparkContext

    val initStartTime = System.nanoTime()

    val codeVectors = initialModel match {
      case Some(initialMap) => initialMap.prototypes.map(new VectorWithNorm(_))
      case None => initRandom(data)
    }
    val dims = codeVectors.head.vector.size

    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    logInfo(f"Initialization took $initTimeInSeconds%.3f seconds.")

    var converged = false
    var cost = 0.0
    var iteration = 0

    val iterationStartTime = System.nanoTime()

    while (iteration < $(maxIter) && !converged) {
      val costAccum = sc.doubleAccumulator
      val bcCodeVectors = sc.broadcast(codeVectors)

      /*
       * Kohonen batch algorithm
       * (M. Lebbah, Thesis, p.26)
       */

      val T = computeTemperature(iteration)

      // Find the sum and count of points mapping to each code vector
      val totalContribs = data.mapPartitions { points =>
        val thisCodeVectors = bcCodeVectors.value
        val dims = thisCodeVectors.head.vector.size

        val sums = Array.fill(thisCodeVectors.length)(Vectors.zeros(dims))
        val counts = Array.fill(thisCodeVectors.length)(0L)

        points.foreach { point =>
          val (bestMatchingUnit, cost) = SOM.findClosest(thisCodeVectors, point)
          costAccum.add(cost)
          val sum = sums(bestMatchingUnit)

          axpy(1.0, point.vector, sum)
          counts(bestMatchingUnit) += 1
        }

        counts.indices.filter(counts(_) > 0).map(j => (j, (sums(j), counts(j)))).iterator
      }.reduceByKey { case ((sum1, count1), (sum2, count2)) =>
        axpy(1.0, sum2, sum1)
        (sum1, count1 + count2)
      }.collectAsMap()

      bcCodeVectors.destroy()

      converged = true

      // Compute the neighborhood-weighted sums and counts
      val weights = Array.tabulate(codeVectors.length) { i =>
        Array.tabulate(codeVectors.length)(j => computeNeighborhood(cellDist(i, j), T))
      }

      val weightedContribs = (0 until codeVectors.length).map { k =>

        val weightedSum = Vectors.zeros(dims)
        var weightedCount = 0D

        totalContribs.foreach { case (j, (sum, count)) =>
            axpy(weights(k)(j), sum, weightedSum)
            weightedCount += weights(k)(j) * count
        }

        (k, (weightedSum, weightedCount))
      }

      // Update the map code vectors and costs
      weightedContribs.foreach { case (k, (sum, count)) =>
        scal(1.0 / count, sum)
        val newCodeVector = new VectorWithNorm(sum)
        if (converged && SOM.fastSquaredDistance(newCodeVector, codeVectors(k)) > $(tol) * $(tol)) {
          converged = false
        }
        codeVectors(k) = newCodeVector
      }

      cost = costAccum.value
      logInfo(s"SOM quantization error: $cost")
      iteration += 1
    }

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
    logInfo(f"Iterations took $iterationTimeInSeconds%.3f seconds.")

    if (iteration == $(maxIter)) {
      logInfo(s"SOM reached the max number of iterations: ${$(maxIter)}.")
    } else {
      logInfo(s"SOM converged in $iteration iterations.")
    }

    logInfo(s"The cost is $cost.")

    new SOMModel(Identifiable.randomUID("SOMModel"), codeVectors.map(_.vector))

  }

  /**
    * Temperature decay function
    */
  private def computeTemperature(iter: Int): Double = $(temperatureDecay) match {
    case "exponential" => $(tMax) * pow( ($(tMin) / $(tMax)), (iter.toDouble / ($(maxIter) - 1) ) )
    case "linear" => $(tMax) + (iter.toDouble / ($(maxIter) - 1)) * ($(tMin) - $(tMax))
  }

  /**
    * Neighborhood kernel function
    */
  private def computeNeighborhood(d: Int, T: Double): Double = $(neighborhoodKernel) match {
    case "gaussian" => exp(-(d*d).toDouble / (T*T))
    case "rectangular" => if (d <= T) 1.0 else 0.0
  }

  /**
    * Manhattan distance between two map cells
    */
  private def cellDist(id1: Int, id2: Int): Int = $(topology) match {
    case "rectangular" => abs(id2 / $(width) - id1 / $(width)) + abs(id2 % $(width) - id1 % $(width))
  }

  /**
    * Initialize a set of code vectors at random.
    */
  private def initRandom(data: RDD[VectorWithNorm]): Array[VectorWithNorm] = {
    // Select with replacement
    data.takeSample(true, $(height) * $(width), new XORShiftRandom($(seed)).nextInt())
  }

}

object SOM {
  /**
    * Returns the index of the closest center to the given point, as well as the squared distance.
    */
  def findClosest(centers: TraversableOnce[VectorWithNorm],
                  point: VectorWithNorm): (Int, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      // Since `\|a - b\| \geq |\|a\| - \|b\||`, we can use this lower bound to avoid unnecessary
      // distance computation.
      var lowerBoundOfSqDist = center.norm - point.norm
      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
      if (lowerBoundOfSqDist < bestDistance) {
        val distance: Double = fastSquaredDistance(center, point)
        if (distance < bestDistance) {
          bestDistance = distance
          bestIndex = i
        }
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  /**
    * Returns the K-means cost of a given point against the given cluster centers.
    */
  def pointCost(centers: TraversableOnce[VectorWithNorm],
                point: VectorWithNorm): Double =
    findClosest(centers, point)._2

  /**
    * Returns the squared Euclidean distance between two vectors computed by
    * [[xyz.florentforest.spark.ml.util.MLUtils#fastSquaredDistance]].
    */
  def fastSquaredDistance(v1: VectorWithNorm, v2: VectorWithNorm): Double = {
    MLUtils.fastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm)
    //Vectors.sqdist(v1.vector, v2.vector)
  }

}

/**
  * A vector with its norm for fast distance computation.
  *
  * @see [[xyz.florentforest.spark.ml.som.SOM#fastSquaredDistance]]
  */
class VectorWithNorm(val vector: Vector, val norm: Double) extends Serializable {

  def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))

  def this(array: Array[Double]) = this(Vectors.dense(array))

  /** Converts the vector to a dense vector. */
  def toDense: VectorWithNorm = new VectorWithNorm(Vectors.dense(vector.toArray), norm)
}


class XORShiftRandom(init: Long) extends JavaRandom(init) {

  def this() = this(System.nanoTime)

  private var seed = XORShiftRandom.hashSeed(init)

  // we need to just override next - this will be called by nextInt, nextDouble,
  // nextGaussian, nextLong, etc.
  override protected def next(bits: Int): Int = {
    var nextSeed = seed ^ (seed << 21)
    nextSeed ^= (nextSeed >>> 35)
    nextSeed ^= (nextSeed << 4)
    seed = nextSeed
    (nextSeed & ((1L << bits) -1)).asInstanceOf[Int]
  }

  override def setSeed(s: Long) {
    seed = XORShiftRandom.hashSeed(s)
  }
}

object XORShiftRandom {

  /** Hash seeds to have 0/1 bits throughout. */
  def hashSeed(seed: Long): Long = {
    val bytes = ByteBuffer.allocate(java.lang.Long.SIZE).putLong(seed).array()
    val lowBits = MurmurHash3.bytesHash(bytes)
    val highBits = MurmurHash3.bytesHash(bytes, lowBits)
    (highBits.toLong << 32) | (lowBits.toLong & 0xFFFFFFFFL)
  }
}

/**
  * Main object for test and benchmark purpose
  */
object Main extends App {

  println("Spark ML SOM test")

  val spark = SparkSession
    .builder()
    .appName("Spark SOM test (xyz.florentforest.spark.ml.som)")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  final val N = 10
  val data = Seq.tabulate(N){ i => (0.0, Vectors.dense(255.0*i/N,255.0*(N-i)/N, 0.0)) }

  val df = data.toDF("label", "features")

  val som: SOM = new SOM() // default params

  val map: SOMModel = som.fit(df)

  val results = map.transform(df)
  results.show()

  println(map.prototypes.length)

  map.prototypes.foreach(println)

  spark.stop()
}