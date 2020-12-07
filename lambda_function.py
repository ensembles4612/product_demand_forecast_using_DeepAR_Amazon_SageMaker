# attention: edit permission -- IAM execution role -- VolumeForecastDeeparLambda-role-h7l0n5ri so that it includes the policy from https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/, which gives your Lambda function permission to invoke a model endpoint.

import os
import io
import boto3
import json
import csv


# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    data = json.dumps(event)

    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, Body=data)
    print(response)
    result = json.loads(response['Body'].read().decode('utf-8'))

    return result
