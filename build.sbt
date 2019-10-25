name := "sparkml-som"
organization := "xyz.florentforest"

version := "0.1"

scalaVersion := "2.11.8"

val sparkVersion = "2.2.0"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
)

// POM settings for Sonatype
organization := "xyz.florentforest"
homepage := Some(url("https://github.com/FlorentF9/sparkml-som"))
scmInfo := Some(ScmInfo(url("https://github.com/FlorentF9/sparkml-som"),
  "git@github.com:FlorentF9/sparkml-som.git"))
developers := List(Developer("FlorentF9",
  "Florent Forest",
  "florent.forest9@gmail.com",
  url("http://florentfo.rest")))
licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0"))
publishMavenStyle := true

// Add sonatype repository settings
publishTo := Some(
  if (isSnapshot.value)
    Opts.resolver.sonatypeSnapshots
  else
    Opts.resolver.sonatypeStaging
)
