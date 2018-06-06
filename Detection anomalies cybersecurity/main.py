from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql import functions 
import pandas as pd
import numpy as np
from pyspark.ml.feature import OneHotEncoder, StringIndexer 
from pyspark.mllib.util import MLUtils
from pyspark.ml.feature import Normalizer
import pyspark.sql.functions as func
from pyspark.sql.functions import udf
from pyspark.mllib.stat import Statistics
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.tree import RandomForest, RandomForestModel
from numpy import array
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler

sc = SparkContext("local", appName="prueba")
sql = SQLContext(sc)


data = sql.read.load('dataprueba_netflow.csv',sep=',',format='csv',header='true', inferSchema='true')

#print "Size:", (data.count(), len(data.columns))

data = data.select('ts', 'te', 'td', 'sa', 'da', 'sp', 'dp', 'pr', 'flg', 'ipkt', 'ibyt')
data.cache()

#print "Types:", (data.printSchema())

data2 = data.orderBy('ts',ascending=True)

data3 = data2.withColumn('td',functions.when(data2['td']==0.0,0.0005).otherwise(data2['td']))


data4 = data3.withColumn('pps', data3['ipkt']/data3['td'])
data5 = data4.withColumn('bps', data3['ibyt']/data3['td'])
data6 = data5.withColumn('bpp', data3['ipkt']/data3['ibyt'])


stringIndexer_pr = StringIndexer(inputCol='pr', outputCol='pr_enc')
model_pr = stringIndexer_pr.fit(data6)
indexed_pr = model_pr.transform(data6)

stringIndexer_flg = StringIndexer(inputCol='flg', outputCol='flg_enc')
model_flg = stringIndexer_flg.fit(indexed_pr)
indexed_flg = model_flg.transform(indexed_pr)

dataenc = indexed_flg.drop('pr', 'flg')


datagroup = dataenc.groupBy(["ts", "sa", "dp" , "pr_enc", "flg_enc"]).sum() 
datagroup = datagroup.drop("sum(ts)", "sum(dp)", "sum(pr_enc)", "sum(flg_enc)", "sum(sp)")
 
ncon = dataenc.groupBy(["ts", "sa", "dp" , "pr_enc", "flg_enc"]).count()
#ncon2 = ncon.withColumn("ncon", ncon["count"].cast("double"))
ncon2 = ncon.select('count')

datagroup=datagroup.withColumn('row_index', func.monotonically_increasing_id())
ncon2=ncon2.withColumn('row_index', func.monotonically_increasing_id())
datacon = datagroup.join(ncon2["row_index","count"], on=["row_index"]).drop("row_index")

#print datacon.show(3)
#print "Size:", (datacon.count(), len(datacon.columns))


datacon2 = datacon.drop('ts', 'sa')
print datacon2.show(10)
print datacon2.printSchema()
print datacon2.columns

parsedData = datacon2.rdd.map(lambda line: array([float(x) for x in line]))
model = KMeans.train(parsedData, 2, maxIterations=10, initializationMode='random')
print model
centers = model.clusterCenters
for center in centers:
	print(center)

assembler = VectorAssembler(
	inputCols=datacon2.columns,
	outputCol="features")

output=assembler.transform(datacon2)


pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
model_pca = pca.fit(output)
datacon_pca = model_pca.transform(output).select("pcaFeatures")
print datacon_pca.show(10, truncate=False)

df_data = datacon_pca.toPandas()
np_data = df_data.values
print np_data[0]
print np_data[1]
print np_data.shape




