'''
Handler code for the different Lambda functions in the workflow
'''

######################
#1. serializeImageData
######################
import json
import boto3
import base64

s3 = boto3.client('s3')


def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    s3 = boto3.client('s3')
    file_name = '/tmp/image.png'
    s3.download_file(bucket, key, file_name)

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

######################
#2. classifyImage
######################
import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2023-01-11-16-23-22-165'

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['image_data'])

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(
        ENDPOINT,
        serializer=sagemaker.serializers.IdentitySerializer()
    )

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)

    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }

######################
#3. filterClassifications
######################
import json

THRESHOLD = .7#Arbitrary value, suggested is .7


def lambda_handler(event, context):

    # Grab the inferences from the event
    inferences = json.loads(event['inferences'])

    # Check if any values in our inferences are above THRESHOLD
    bicycle_score = float(inferences[0])
    motorcycle_score = float(inferences[1])
    meets_threshold = bicycle_score > THRESHOLD or motorcycle_score > THRESHOLD
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise BaseException("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }