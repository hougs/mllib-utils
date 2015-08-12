package com.cloudera.datascience

import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithSGD}
import org.apache.spark.mllib.optimization._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD


object RegularizedLogisticRegression {
  def run(trainingData: RDD[LabeledPoint], regularization: String, stepSize: Option[Double],
          regParam: Option[Double]):  LogisticRegressionModel = {
    val algorithm = new LogisticRegressionWithSGD()
    /** Create a trainable model that uses the specified type of regularization. */
    val updater = regularization match {
      case "L1" => new L1Updater
      case "L2" => new SquaredL2Updater
      case "none" => new SimpleUpdater
    }
    algorithm.optimizer.setUpdater(updater)
    stepSize.fold()(algorithm.optimizer.setStepSize(_))
    regParam.fold()(algorithm.optimizer.setRegParam(_))
    algorithm.run(trainingData)
  }
}
