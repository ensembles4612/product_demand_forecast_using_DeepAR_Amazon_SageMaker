# product_demand_forecast_using_DeepAR_Amazon_SageMaker
## Project Highlights 

## Project Object

A company has a large portfolio of products distributed to retailers through agencies. There are thousands of unique agency-SKU/product combinations. In order to plan its production and distribution as well as help agencies with their planning, it is important for the company to have an accurate estimate of demand at SKU level for each agency.
Our purpose is to achieve the following using DeepAR algorithm:
* taking advantage of DeepAR's RNN framework. To predict product demand that's not only trained on their own historical data, but also trained on historical data of other variables(e.g., sales price and promotion price) that have impact on product demand.
* predicting the monthly demand volume(hectoliters) for each Agency-SKU combination (350 Agency-SKU combinations in total) for year 2018 based on data from 2013-2017.
* handling cold start problems, which is to do volume demand forecast for new Agency-SKU combinations that we don't have any data on.

## Data Extraction and transformation
## EDA
## Model Training and Fine-tuning
## Model Performance

## Productionization

I invoked the model endpoint deployed by Amazon SageMaker using API Gateway and AWS Lambda. For testing purposes, I used Postman. 

How it works: starting from the client side, a client script calls an Amazon API Gateway API action and passes parameter values. API Gateway is a layer that provides API to the client. In addition, it seals the backend so that AWS Lambda stays and executes in a protected private network. API Gateway passes the parameter values to the Lambda function. The Lambda function parses the value and sends it to the SageMaker model endpoint. The model performs the prediction and returns the predicted value to AWS Lambda. The Lambda function parses the returned value and sends it back to API Gateway. API Gateway responds to the client with that value.


## Code and Resources 

**Python Version:** 3.7  
**Packages:** pandas, numpy, matplotlib,flask, json  
**Deployment reference:** https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/ 
**Data Source:** https://www.kaggle.com/utathya/future-volume-prediction
