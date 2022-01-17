from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from PIL import Image
import numpy as np
import math
import findspark
import os
findspark.init()

# conf = SparkConf().setAppName('appName').setMaster("local")
# sc = SparkContext(conf=conf)

conf = SparkConf().setAppName('coreset').setMaster("local")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

k = 2
d = 3
max_iterations = 10

def get_image_array(image_name):
    image = Image.open(image_name)
    pix = image.load()
    width = image.size[0]
    height = image.size[1]

    return pix, width, height



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

def create_coreset_segmentation(input_name, output_name, eps, a):
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
                                dct[ri, gi, bi] = np.array([dct[ri, gi, bi][0] + sum(
                                    points[(np.where((points == point).all(axis=1))[0])][:, 0]), point[1], point[2],
                                                            point[3]])
                            except KeyError:
                                dct[ri, gi, bi] = np.array(
                                    [sum(points[(np.where((points == point).all(axis=1))[0])][:, 0]), point[1],
                                     point[2], point[3]])

                            points = np.delete(points, np.where((points == point).all(axis=1)), 0)

                result.extend(list(dct.values()))
                j += 1

        return (round(key % n_new), result)

    # get data
    image_array, width, height = get_image_array(input_name)
    data = []
    for x in range(width):
        for y in range(height):
            rgb = image_array[x, y]
            r = rgb[0]
            g = rgb[1]
            b = rgb[2]
            data.append((x, y, r, g, b, Vectors.dense(float(r), float(g), float(b))))

    # Add index and weight to data
    df = spark.createDataFrame(data, ["x", "y", "r", "g", "b", "features"])
    zipped = df.rdd.zipWithIndex()
    df = zipped.map(lambda x: [x[1], x[0], 1]).toDF(["index", 'features2', 'weighCol'])
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
    model = kmeans.fit(df)
    model.predict(df.head().features)
    centers = model.clusterCenters()

    summary = model.summary

    # Calculate important variables
    n = len(data)
    z = np.log(n) * np.log(a * np.log(n))
    r = (summary.trainingCost / (a * np.log(n) * n))**0.5
    x_c = eps * r / d**0.5

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

    # Do kmeans on coreset
    S_core = []
    for i in c[0][1]:
        S_core.append((float(i[0]), Vectors.dense(i[1], i[2], i[3])))

    df_core = spark.createDataFrame(S_core, ['weighCol', 'features'])
    kmeans = KMeans(k=k)
    kmeans.setSeed(1)
    kmeans.setWeightCol("weighCol")
    kmeans.setMaxIter(max_iterations)
    model = kmeans.fit(df_core)
    model.predict(df.head().features)
    centers_core = model.clusterCenters()
    summary_core = model.summary
    rows = model.transform(df).select("x", "y", "r", "g", "b", "prediction")

    # get image segmentation using coreset
    value = np.empty((), dtype='uint8, uint8, uint8')
    value[()] = (0, 0, 0)
    compressed_array = np.full((height, width), value, dtype='uint8, uint8, uint8')

    colors = {}
    for i in range(min(len(S_core), k)):
        s = rows.filter(col("prediction") == i).rdd.first()

        # Set RGB values as a tuple
        colors[i] = (s[2], s[3], s[4])

    row_data = rows.collect()
    for i in range(len(row_data)):
        row = row_data[i]
        color = colors[row.prediction]
        compressed_array[int(row.y), int(row.x)] = color


    print(output_name, colors)
    # image_from_rgb = Image.fromarray(compressed_array, 'RGB')
    # image_from_rgb.save(output_name)


directory = os.fsencode('C:\\Users\\20202958\\source\\repos\\kmeansplus\\Fruit')
eps_ranges = [1]
a_ranges = [1, 10, 100] #try 0.1 later

for eps in eps_ranges:
    for a in a_ranges:

        folder = os.path.join('C:\\Users\\20202958\\source\\repos\\kmeansplus\\Fruit', 'coreset', 'eps' + str(eps) + 'a' + str(a))
        # os.mkdir(folder)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                create_coreset_segmentation(os.path.join('C:\\Users\\20202958\\source\\repos\\kmeansplus\\Fruit', filename), os.path.join(folder, filename), eps, a)
