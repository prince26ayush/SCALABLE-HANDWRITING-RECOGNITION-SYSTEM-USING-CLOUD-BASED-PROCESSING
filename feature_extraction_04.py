import cv2
import numpy as np
import boto3
import logging
from pyspark import SparkContext

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_hog_features(s3_path):
    """
    Extracts HOG features from an image stored in S3.

    Args:
        s3_path (str): The S3 path of the image (e.g., "s3://bucket/path/to/image.png").

    Returns:
        tuple: A tuple containing the S3 path and the HOG features as a list, or None if an error occurred.
    """
    try:
        # Extract bucket and key from S3 path
        s3_parts = s3_path.replace("s3://", "").split("/")
        bucket_name = s3_parts[0]
        key = "/".join(s3_parts[1:])

        # Download image from S3
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket_name, Key=key)
        image_data = response['Body'].read()

        # Decode image using OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            logging.error(f"Error decoding image: {s3_path}")
            return None

        # Compute HOG features
        hog = cv2.HOGDescriptor()
        features = hog.compute(img)

        # Flatten features and convert to list
        features_list = features.flatten().tolist()

        logging.info(f"HOG features extracted for: {s3_path}, feature length: {len(features_list)}")
        return s3_path, features_list

    except Exception as e:
        logging.error(f"Error processing image {s3_path}: {e}")
        return None

if __name__ == "__main__":
    sc = SparkContext(appName="HOGFeatureExtraction")

    # Configure S3 bucket and prefix (adjust these to your actual values)
    s3_bucket = "handwriting-recognition-yourinitials" #The bucket where the preprocessed images are stored
    s3_prefix = "preprocessed/" #The folder in your bucket where the preprocessed images are stored

    # List all image paths in the S3 bucket and prefix
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(s3_bucket)
    image_paths = [f"s3://{s3_bucket}/{obj.key}" for obj in bucket.objects.filter(Prefix=s3_prefix)]

    # Parallelize image paths using Spark RDD
    rdd = sc.parallelize(image_paths)

    # Extract HOG features for each image
    results = rdd.map(extract_hog_features).filter(lambda x: x is not None)

    # Collect and print the results (for demonstration)
    for s3_path, features in results.collect():
        print(f"Features for {s3_path}: {len(features)} elements")

    # Optionally, write the results to a file or S3
    # Example:
    results.saveAsTextFile("s3://handwriting-recognition-yourinitials/hog_features")

    sc.stop()