# Product Demand Forecast Using DeepAR Algorithm in SageMaker AWS

## Project Highlights 



## Project Object

The company has a large portfolio of beer products distributed to retailers through agencies. There are thousands of unique agency-SKU/product combinations. In order to plan its production and distribution as well as help agencies with their planning, it is important for the company to have an accurate estimate of monthly demand at SKU level for each agency.

Our purpose is to achieve the following using DeepAR algorithm:
* taking advantage of DeepAR's RNN framework. To predict product demand that's not only trained on their own historical data, but also trained on historical data of other variables(e.g., on-sale price and promotion price) that have impact on product demand.
* predicting the monthly demand volume(hectoliters) for each Agency-SKU combination (350 Agency-SKU combinations in total) for year 2018 based on data from 2013-2017.
* handling cold start problems, which is to forecast volume demand for new Agency-SKU combinations that we don't have any data on.

## Why use Amazon Sagemaker DeepAR Algorithm

## Sagemaker Instance Setup

After downloading the data from Kaggle and uploading them to aws S3 bucket, I set up an instance in sagemaker with an IAM role created where I created a Python SDK notebook. In the notebook, I connected the notebook instance with the data in S3 bucket using IAM role, and configured the DeepAR container image to be used for the region that I ran in.  

## Data Cleaning

After the instance setup, I did some data cleaning, transforming the dataset from csv format to json lines with certain requirements so DeepAR algorithm is able to process it. The data originally is tabular from Jan 2013 to Dec 2017 with 7 columns:
* Agency (agency that sells SKUs, e.g., Agency_01)
* SKU (a certain beer product, e.g., SKU_01)
* YearMonth (yyyymm, e.g., 201701)
* Volume (actual volume sales, hectoliters), 
* Price (regular price, $/hectoliters), 
* Sales (on-sale price,$/hectoliters)
* Promotions (Promotions = Price -Sales, $/hectoliter)

The main data cleaning I did:
* loaded the datasets from S3 bucket and joined them
* encoded categorical features -- Agency and SKU to numbers (e.g. Agency_01 and SKU_01 are now [0,0]) required by DeepAR
* changed variables -- Sales and Promotions to dynamic feature format required by DeepAR so they can help predict demand volume for each Agency-SKU combination 
* splitted data to training and test sets: 
  * training set: 48-month data from 2013-01 to 2016-12
  * test set: 12-month data from 2017-01 to 2017-12
* transformed the tabular dataset to dictionary format
* transformed the dictionary format to json lines format which is then uploaded to S3 bucket

After cleaning and transforming data to jsonlines, for example, the first 2 lines in test set look like:
`{"start": "2013-01-01 00:00:00", "target": [80.676, 98.064, 133.704, ..., 37.908, 35.532], "cat": [0, 0], "dynamic_feat": [[1033.432731, 1065.417195, 1101.133633, ..., 1341.864851, 1390.112272], [108.067269, 76.08280500000001, 78.212187, ..., 281.01264199999997, 273.68522]]}`

`{"start": "2013-01-01 00:00:00", "target": [78.408, 99.25200000000001, 137.268, ..., 24.191999999999997, 17.172], "cat": [0, 1], "dynamic_feat": [[969.1862085000001, 996.9507620999, 1061.272227, ..., 1351.3808589999999, 1412.680031], [104.9715905, 77.99408290000001, 67.71759399, ..., 321.32673, 284.895441]]}`

## EDA using Tableau
I made an interactive dashboard regarding product demand, on-sale price and promotion trend over 2013-2017. It can be found in Tableau Public under my profile: https://public.tableau.com/profile/shelley8110#!/vizhome/productdemandDashboard/InteractiveDashboard?publish=yes

## Model Training and Fine-tuning

1. After the initial training with specific hyperparameter setting below, I obtained test RMSE: <loss>=759.711203642
```md  
  hyperparameters = {
    "time_freq":"M" ,
    "cardinality" : "auto",
    "num_dynamic_feat":"auto",
    "epochs": "162",       
    "dropout_rate": "0.1",
    "embedding_dimension" : "10",
    "likelihood":"gaussian",
    "num_cells": "94",
    "num_layers": "4",
    "mini_batch_size":"256",
    #"learning_rate": "1e-3",
    "num_eval_samples": "100",
    "test_quantiles": "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]",
    "context_length": "12",
    "prediction_length": "12"}
```
2. I trained 6 auto hyperparameter tuning jobs to search among specific ranges for some parameters below. The best job:test RMSE: 792.3036. We can adjust the ranges in the future to get lower test error.
```md
  hyperparameter_ranges = {"learning_rate": ContinuousParameter(1e-4, 1e-1),
                         "mini_batch_size": IntegerParameter(128, 350),
                         "epochs": IntegerParameter(10, 500),
                         "context_length":IntegerParameter(12, 24),
                         "num_cells":IntegerParameter(30, 200),
                         "num_layers": IntegerParameter(1, 8)
                        }
```
3. I also explored to train 2 more models using Auto-tuning with a warm_start from the previous step. The test errors of the 2 models were high but we can fine tune the model more in the future to obtain better results. For the project at this stage, I used the model with the lowest test RMSE for now. 
 
## Model Performance

* Plotting actual vs. predicted demand volume in test set

I plotted 20 graphs for 20 Agency-SKU combinations below. Each graph has time(month/2017) as x-axis and demand volume(hectoliters) as y-axis. The black line is actual volume and the red line is the predicted volume averaged by all samples, which is the mean prediction. Mean prediction is only a point estimate, and we can see some predicted values can't catch the trend.

![alt text](https://github.com/ensembles4612/product_demand_forecast_using_DeepAR_Amazon_SageMaker/blob/master/Plot%20of%20pred%20vs.%20actual%20volume%20demand%20in%202017%2050-70.png "pred vs actual")

* Adding 10%-90% quantiles to the actual-pred plot
I obtained predicted 10% and 90% quantiles using batch transform, so I could add the 80% interval to the plot to better catch the trend of ther predicted demand. We can see below that the 80% intervals of most plots catch the general trend of the actual values. An interval like this can help better prepare for the future volume demand for each Agency-SKU combination.

![alt text](https://github.com/ensembles4612/product_demand_forecast_using_DeepAR_Amazon_SageMaker/blob/master/Plot%20of%20pred%20vs.%20actual%20volume%20demand%20in%202017%2050-70%20with%2080%25%20interval.png "pred vs actual with interval")

## Productionization

I invoked the model endpoint deployed by Amazon SageMaker using API Gateway and AWS Lambda. For testing purposes, I used Postman. 

How it works: starting from the client side, a client script calls an Amazon API Gateway API action and passes parameter values. API Gateway is a layer that provides API to the client. In addition, it seals the backend so that AWS Lambda stays and executes in a protected private network. API Gateway passes the parameter values to the Lambda function. The Lambda function parses the value and sends it to the SageMaker model endpoint. The model performs the prediction and returns the predicted value to AWS Lambda. The Lambda function parses the returned value and sends it back to API Gateway. API Gateway responds to the client with that value.



## References

**Python Version:** 3.7  
**Packages:** pandas, numpy, matplotlib,flask, json  
**Sagemaker Endpiont Deployment reference:** https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/ 
**Dataset Source:** https://www.kaggle.com/utathya/future-volume-prediction
**DeepAR Paper:** https://arxiv.org/pdf/1704.04110v3.pdf
**Other DeepAR Article:** https://aws.amazon.com/blogs/machine-learning/now-available-in-amazon-sagemaker-deepar-algorithm-for-more-accurate-time-series-forecasting/

