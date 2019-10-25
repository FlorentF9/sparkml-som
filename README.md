# Spark ML SOM (Self-Organizing Map)

SparkML-SOM is the only available distributed implementation of Kohonen's Self-Organizing-Map algorithm built on top of Spark ML (the Dataset-based API of Spark MLlib) and fully compatible with Spark versions 2.2.0 and newer. It extends Spark's [`Estimator`](https://github.com/apache/spark/blob/v2.2.0/mllib/src/main/scala/org/apache/spark/ml/Estimator.scala) and [`Model`](https://github.com/apache/spark/blob/v2.2.0/mllib/src/main/scala/org/apache/spark/ml/Model.scala) classes.

* SparkML-SOM can be used as any other MLlib algorithm with a simple `fit` + `transform` syntax
* It is compatible with Datasets/DataFrames
* It can be integrated in a Spark ML Pipeline
* It leverages fast native linear algebra with BLAS

The implemented algorithm is the Kohonen batch algorithm, which is very close to the $k$-means algorithm, but the computation of the average code vector is replaced with a topology-preserving weighted average. For this reason, most of the code is identical to MLlib's $k$-means implementation (see [`org.apache.spark.ml.clustering.KMeans`](https://github.com/apache/spark/blob/v2.2.0/mllib/src/main/scala/org/apache/spark/ml/clustering/KMeans.scala) and [`org.apache.spark.mllib.clustering.KMeans`](https://github.com/apache/spark/blob/v2.2.0/mllib/src/main/scala/org/apache/spark/mllib/clustering/KMeans.scala)).

The same algorithm was implemented by one of my colleagues: https://github.com/TugdualSarazin/spark-clustering (project now maintained by [C4E](https://github.com/Clustering4Ever/Clustering4Ever)).
This version is meant to be simpler to use and more concise, performant and compatible with Spark ML Pipelines and Datasets/DataFrames.

**This code will soon be integrated into the [C4E clustering project](https://github.com/Clustering4Ever/Clustering4Ever)**, so be sure to check out this project if you want to explore more clustering algorithms. In case you only need SOM, keep using this code which will remain independant and up-to-date.

## Quickstart

```scala
import xyz.florentforest.spark.ml.som.SOM

val data: DataFrame = ???

val som = new SOM()
    .setHeight(20)
    .setWidth(20)

val model = som.fit(data)

val res: DataFrame = model.transform(data)
```

## Installation

The quickest way to use this package is to clone the repository, compile it using sbt and publishing it locally:

```shell
$ git clone git@github.com:FlorentF9/sparkml-som.git
$ cd sparkml-som
$ sbt publishLocal
```

Then, use it in your projects by adding the dependency line to sbt:

```sbt
"xyz.florentforest" %% "sparkml-som" % "0.1"
```

## Parameters

Self-organizing maps essentially depend on their topology, the neighborhood function and the neighborhood radius decay. The algorithm uses a temperature parameter that decays after each iteration and controls the neighborhood radius. It starts at a value $T_{max}$ that should cover the entire map and decreases to a value $T_{min}$ that should cover a single map cell. Here are the configuration parameters:

* **Map grid topology** (`topology`)
  * rectangular _(default)_
* **Height and width**: `height` _(default=10)_, `width`_(default=10)_
* **Neighborhood kernel** (`neighborhoodKernel`)
  * gaussian _(default)_
  * rectangular window
* **Temperature (or radius) decay** (`temperatureDecay`)
  * exponential _(default)_
  * linear
* **Initial and final temperatures**: `tMax` _(default=10.0)_, `tMin` _(default=1.0)_
* **Maximum number of iterations**: `maxIter` _(default=20)_
* **Tolerance (for convergence)**: `tol` _(default=1e-4)_

## Implementation details

The package depends only on spark (core, sql and mllib) and netlib for native linear algebra. It will use native BLAS libraries if possible. Because of classes and methods marked as private in spark, some utility and linear algebra code from spark had to be included into the project: _util.SchemaUtils_, _util.MLUtils_ and _linalg.BLAS_. I kept the original license and tried to keep the code minimal with only the parts needed by SOM.

## To-dos

* Add hexagonal grid topology
* Add visualization capabilities
* I did not extend MLWritable/MLReadable yet, so the model cannot be saved or loaded. However, as all the parameteres are stored in the `SOMModel.prototypes` variable of type `Array[Vector]`, it is straightforward to save the parameters into a file.
