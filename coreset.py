from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from PIL import Image
import numpy as np
import math
import findspark
findspark.init()

# conf = SparkConf().setAppName('appName').setMaster("local")
# sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

k = 2
d = 3
a = 1
eps = 0.1
max_iterations = 10

def get_image_array():
    image = Image.open('test-small.PNG')
    pix = image.load()
    width = image.size[0]
    height = image.size[1]

    return pix, width, height

# get data
image_array, width, height = get_image_array()
data = []
for x in range(width):
    for y in range(height):
        rgb = image_array[x, y]
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
        data.append((x, y, r, g, b, Vectors.dense(int(r), int(g), int(b))))

# Add index and weight to data
df = spark.createDataFrame(data, ["x", "y", "r", "g", "b", "features"])
df = df.rdd.zipWithIndex().map(lambda x: [x[1], x[0], 1]).toDF(["index", 'features2', 'weighCol'])
df = df.withColumn('x', df['features2'].getItem("x"))
df = df.withColumn('y', df['features2'].getItem("y"))
df = df.withColumn('r', df['features2'].getItem("r"))
df = df.withColumn('b', df['features2'].getItem("b"))
df = df.withColumn('g', df['features2'].getItem("g"))
df = df.withColumn('features', df['features2'].getItem("features"))
df = df.drop("features2")
df.printSchema()

# Run the kmeans++ algorithm
kmeans = KMeans(k=k)
kmeans.setSeed(1)
kmeans.setWeightCol("weighCol")
kmeans.setMaxIter(max_iterations)
kmeans.clear(kmeans.maxIter)
model = kmeans.fit(df)
model.predict(df.head().features)
centers = model.clusterCenters()

summary = model.summary
print(summary.clusterSizes, summary.trainingCost, centers)

# Calculate important variables
n = len(data)
z = np.log(n) * np.log(a * np.log(n))
r = (summary.trainingCost / (a * np.log(n) * n))**0.5
x_c = eps * r / d**0.5
print(z, r)

print(2*r/x_c)

def coreset(points):
    result = []
    key = points[0]
    points = np.array(points[1])
    j = 0

    while j < round(z) and len(points) > 0:
        x_j = (eps / 4 * i) * 2 ** j * r / d ** 0.5
        for centroid in centers:
            # Create the grids
            grid_r = np.linspace(centroid[0] - 2 ** j * r, centroid[0] + 2 ** j * r, round(2 * 2 ** j * r / x_j))
            grid_g = np.linspace(centroid[1] - 2 ** j * r, centroid[1] + 2 ** j * r, round(2 * 2 ** j * r / x_j))
            grid_b = np.linspace(centroid[1] - 2 ** j * r, centroid[1] + 2 ** j * r, round(2 * 2 ** j * r / x_j))
            dct = dict()
            # Check if a point lays within the circle and see in which grid block it lays
            for point in points:
                if np.linalg.norm(centroid - point[1:]) <= 2 ** j * r:
                    ri = False
                    gi = False
                    bi = False
                    for ris in grid_r:
                        if point[1] >= ris:
                            ri = ris
                            break
                    for gis in grid_g:
                        if point[2] >= gis:
                            gi = gis
                            break
                    for bis in grid_b:
                        if point[3] >= bis:
                            bi = bis
                            break

                    # If the point lays within the circle and within a gridcell than add them to the results.
                    # See if there are duplicates of this point. Add all the weights of the duplicates and if there
                    # was already a point in this gridcell then add these weights too. Delete the point from points.
                    if ri and gi and bi:
                        try:
                            dct[ri, gi, bi] = np.array([dct[ri, gi, bi][0] + point[0] * len(np.where((points == point).all(axis=1))[0]), point[1], point[2], point[3]])
                        except KeyError:
                            dct[ri, gi, bi] = np.array([point[0] * len(np.where((points == point).all(axis=1))[0]), point[1], point[2], point[3]])

                        points = np.delete(points, np.where((points == point).all(axis=1)), 0)

            result.extend(list(dct.values()))
            j += 1

    return (round(key % n_new), result)


def combineValues1(range, value):
    newRange = []
    for item in range:
        newRange.append(item)

    newRange.append(value)

    return newRange


def combineValues2(range, value):
    newRange = []
    for item in range:
        newRange.append(item)

    newRange.append(value[0])

    return newRange

# Devide the problem equally over sqrt(n) machines, and calculate on each machine the coreset.
i = 1
rdd = df.rdd.map(lambda x: (round(x[0] % (n ** 0.5)), [x[1], x[4], x[5], x[6]]))
n_new = n ** 0.5
n_new = math.ceil(n_new / 2)
rdd = rdd.combineByKey(lambda a: [a], combineValues1, lambda a, b: a + b)
rdd = rdd.map(coreset)
c = rdd.collect()

# Merge the coresets.
while len(c) > 1:
    rdd = rdd.combineByKey(lambda a: a, combineValues2, lambda a, b: a + b)
    n_new = math.ceil(n_new / 2)
    print(n_new)
    i += 1
    rdd = rdd.map(coreset)
    c = rdd.collect()
print(c)