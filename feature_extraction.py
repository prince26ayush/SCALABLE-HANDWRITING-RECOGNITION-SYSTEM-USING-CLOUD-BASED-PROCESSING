import cv2
import boto3
import numpy as np
from pyspark.sql import SparkSession
import logging
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_hog_features(image_data):
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.error("Error decoding image.")
            return None

        hog = cv2.HOGDescriptor()
        features = hog.compute(img)
        return features.flatten().tolist()
    except Exception as e:
        logging.error(f"Error extracting HOG features: {e}")
        return None

def process_s3_image(s3_path):
    try:
        s3 = boto3.client('s3')
        bucket, key = s3_path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        image_data = obj['Body'].read()
        features = extract_hog_features(image_data)

        if features:
            # Save features to a CSV-like string
            feature_string = ",".join(map(str, features))
            #save the features to s3
            s3.put_object(Bucket=bucket, Key="features/" + key.split("/")[-1].replace(".png", ".csv"), Body=feature_string)
            logging.info(f"Features extracted and uploaded: {s3_path}")
        else:
            logging.warning(f"Feature extraction failed for: {s3_path}")
    except Exception as e:
        logging.error(f"Error processing {s3_path}: {e}")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("FeatureExtraction").getOrCreate()

    s3 = boto3.client('s3')
    bucket_name = "handwriting-recognition-yourinitials"  # Replace
    prefix = "preprocessed/"  # Process preprocessed images
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    s3_paths = []
    if 'Contents' in objects:
        for obj in objects['Contents']:
            s3_paths.append(f"s3://{bucket_name}/{obj['Key']}")

    rdd = spark.sparkContext.parallelize(s3_paths)
    rdd.foreach(process_s3_image)
    spark.stop()