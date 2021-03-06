from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans
import numpy as np
from PIL import Image
import findspark
import os
findspark.init()

k = 2
max_iterations = 10
conf = SparkConf().setAppName('kmeans').setMaster("local")
sc = SparkContext(conf=conf)


def get_image_array(image_name):
    image = Image.open(image_name)
    pix = image.load()
    width = image.size[0]
    height = image.size[1]

    return pix, width, height


def build_model(data):
    spark = SparkSession.builder.getOrCreate()

    df = spark.createDataFrame(data, ["x", "y", "r", "g", "b", "features"])
    kmeans = KMeans(k=k)
    kmeans.setSeed(1)

    kmeans.setMaxIter(max_iterations)

    model = kmeans.fit(df)

    model.setPredictionCol("prediction")

    model.predict(df.head().features)

    transformed = model.transform(df).select("x", "y", "r", "g", "b", "prediction")
    return transformed


def create_segmentation(input_name, output_name):
    image_array, width, height = get_image_array(input_name)
    data = []
    for x in range(width):
        for y in range(height):
            rgb = image_array[x, y]
            r = rgb[0]
            g = rgb[1]
            b = rgb[2]
            data.append((x, y, r, g, b, Vectors.dense(float(r), float(g), float(b))))

    rows = build_model(data)

    value = np.empty((), dtype='uint8, uint8, uint8')
    value[()] = (0, 0, 0)
    compressed_array = np.full((height, width), value, dtype='uint8, uint8, uint8')

    colors = {}
    for i in range(k):
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
    # image_from_rgb.save(output_name, 'JPEG')


directory = os.fsencode('C:\\Users\\20202958\\source\\repos\\kmeansplus\\Fruit')

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        create_segmentation(os.path.join('C:\\Users\\20202958\\source\\repos\\kmeansplus\\Fruit', filename), os.path.join('C:\\Users\\20202958\\source\\repos\\kmeansplus\\Fruit', 'kmeans', filename))


