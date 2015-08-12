package com.cloudera.datascience

import org.apache.spark.mllib.classification.ClassificationModel
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics, RegressionMetrics}
import org.apache.spark.mllib.regression.{LabeledPoint, RegressionModel}
import org.apache.spark.rdd.RDD

object EvaluationUtils {

  def multiClassMetrics(model: ClassificationModel,
                        testData: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = testData.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }

  def binaryClassMetrics(model: ClassificationModel,
                         testData: RDD[LabeledPoint]): BinaryClassificationMetrics = {
    val predictionsAndLabels = testData.map(example =>
      (model.predict(example.features), example.label)
    )
    new BinaryClassificationMetrics(predictionsAndLabels)
  }

  def regressionMetrics(model: RegressionModel,
                        testData: RDD[LabeledPoint]): RegressionMetrics = {
    val predictionsAndValues = testData.map(example =>
      (model.predict(example.features), example.label)
    )
    new RegressionMetrics(predictionsAndValues)
  }

  def formattedResult[T](metrics: T, params: Map[String, _], returnHeader: Boolean): String = {
    val (metricHeader, metricValues) = metrics match {
      case binary : BinaryClassificationMetrics => binaryMetrics(binary)
    }

    val paramHeader = params.keys.mkString(" , ")
    val paramValues = params.values.map(_.toString).mkString(" , ")
    val valueStr = paramValues + " , " + metricValues
    val headerStr = paramHeader + " , " + metricHeader
    val returnStr = if (returnHeader) headerStr + "\n" + valueStr else valueStr
    return returnStr
  }

  def binaryMetrics(metrics: BinaryClassificationMetrics): (String, String) = {
    val header = List("AUROC", "AUPR")
    val values = List(metrics.areaUnderROC(), metrics.areaUnderPR()).map(_.toString)
    (header.mkString(" , "), values.mkString(" , "))
  }
}
