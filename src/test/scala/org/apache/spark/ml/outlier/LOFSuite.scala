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

package org.apache.spark.ml.outlier

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.LongType
import org.apache.spark.sql.functions._

object LOFSuite {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("LOFExample")
      .master("local[4]")
      .getOrCreate()

    import spark.implicits._
    val df: DataFrame = Seq(
      (1, 1, 2, 2, 1),
      (2, 9, 3, 3, 2),
      (3, 120, 3, 4, 1),
      (4, 25, 2, 5, 2),
      (5, 12, 3, 6, 2),
      (6, 20, 2, 7, 1),
      (7, 1, 2, 2, 1),
      (8, 9, 3, 3, 2),
      (9, 120, 3, 4, 1),
      (10, 25, 2, 5, 2),
      (11, 12, 3, 6, 2),
      (12, 20, 2, 7, 1),
      (13, 1, 2, 2, 1),
      (14, 9, 3, 3, 2),
      (15, 120, 3, 4, 1),
      (16, 25, 2, 5, 2),
      (17, 12, 3, 6, 2),
      (18, 20, 2, 7, 1)
    ).toDF("idx", "a", "b", "c", "target")
    val longDf = df.withColumn("index",col("idx").cast(LongType))

    val assembler = new VectorAssembler()
      .setInputCols(Array("a", "b", "c"))
      .setOutputCol("features")
    val data = assembler.transform(longDf).repartition(4)
    println("data")
    data.show()

    val startTime = System.currentTimeMillis()
    val result = new LOF()
      .setMinPts(5)
      .transform(data)
    val endTime = System.currentTimeMillis()
    result.count()
    result.show()

    //     Outliers have much higher LOF value than normal data
    result.sort(desc(LOF.lof)).head(10).foreach { row =>
      println(row.get(0) + " | " + row.get(1) + " | " + row.get(2))
    }
    println("Total time = " + (endTime - startTime) / 1000.0 + "s")
  }
}
