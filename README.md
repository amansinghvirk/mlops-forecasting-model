# Introduction 

Machine Learning Operations (MLOps) constitutes a series of processes designed to facilitate rapid experimentation, efficient experimentation tracking, the utilization of isolated environments, and the development of production-ready code. It aligns with the application of DevOps principles to the field of Machine Learning Models, emphasizing the seamless transition from development to production. 

The specific focus of this project centers around the Kaggle competition titled "Store Sales Time Series Forecasting," accessible through the following link: [Kaggle - Store Sales Forecating](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). While the competition primarily evaluates prediction quality using the Root Mean Log Squared Error metric, the emphasis of our project diverges. Rather than solely concentrating on prediction accuracy, our goal is to construct a versatile machine learning system capable of accommodating and validating multiple models based on diverse evaluation criteria.

In essence, the envisioned machine learning system aims to provide a framework for conducting and assessing various experiments. This involves running multiple models with distinct hyperparameters and configurations. The system will facilitate a comprehensive evaluation process, allowing for the comparison of models based on a range of training and validation prediction metrics. By doing so, it seeks to go beyond a singular evaluation metric and enable a more nuanced understanding of model performance under different conditions.

Key functionalities of the proposed system include:

- Experimentation Support: The system will support the execution of multiple experiments, each representing a unique model configuration or algorithmic approach.

- Evaluation Criteria Customization: Users will have the flexibility to define and customize evaluation criteria based on the specific requirements of the problem at hand.

- Model Selection: The system will intelligently analyze and compare the results of various experiments, aiding in the automatic selection of the most effective model based on the specified evaluation metrics.

- Scalability and Adaptability: The architecture of the system will be designed with scalability and adaptability in mind, ensuring its applicability not only to the current store sales forecasting challenge but also to a broader spectrum of machine learning problems.

# Solution Approach
The proposed solution approach is designed to shift the focus from singular model accuracy to the development of a comprehensive end-to-end machine learning project. The solution is structured into distinct components, each contributing to the seamless deployment and potential modularization of the entire lifecycle. The major components of the project include:

## SQLite Database:

Utilized for data ingestion, the SQLite database serves as a central repository for storing data crucial for both model training and inference.

## Models:

- Single Store Type Filtering:

    - Data Pipelines: The project initiates by filtering the data for a single store type, streamlining subsequent processing.
    - Training Pipelines: This component encompasses the development and training of machine learning models tailored for the specific store type.
    - Inference Pipelines: Once trained, models are integrated into inference pipelines for making predictions.


## Experiments API:

- An API is established to facilitate experimentation, providing endpoints for accessing diverse experiments, associated parameters, and predictions on both training and validation datasets.

## Deployed API:

- This API serves as an endpoint for predictions using the deployed model, offering a practical solution for real-time forecasting.

# Data Description

The dataset for this project encompasses multiple components, each offering valuable insights into the transactional sales of thousands of products across various store types, considering factors such as holidays and promotions. The duration covered by the dataset spans from January 1, 2013, to August 15, 2017, capturing a comprehensive timeline of sales activities.

## **Key Components of the Dataset**:
<p>&nbsp</p>
- Transactional Sales Data:
    - Provides detailed information on individual transactions, including product details, store types, and corresponding sales figures.
    - Enables the analysis of sales trends, patterns, and variations over time.

- Product Information:
    - Contains details about the diverse range of products within the dataset.
    - May include attributes such as product categories, descriptions, and other relevant information.

- Store Types Information:

    - Describes the various types of stores present in the dataset.
    - Differentiates between stores based on characteristics that could influence sales patterns.

- Holiday Data:

    - Indicates the occurrence of holidays during the specified timeframe.
    - Enables the exploration of potential correlations between holiday periods and sales fluctuations.

- Promotion Data:

    - Provides insights into promotional activities carried out during the specified period.
    - May include details on the types of promotions, duration, and their impact on sales.

# Getting Started
Follow the process for installation and creating the execution enviornment

1. Create virtual enviornment to execute the training code

    ```sh

    # create virtual enviornment
    $ python -m venv .venv

    # activate the virtual enviornment
    $ source .venv/Scripts/activate

    # intall the dependencies
    $ pip install -r requirements.txt

    ```

2. For sample execution use the sample data available in sampledb
    - Create .env file in root folder and add the following contents to the file
    SQLITEDB_PATH="sampledb/stores_sales_sample.db"

    Above path is referring to the link to the database

3. Execute the model training for different experiments

    - below code will output the arguments require for executing the model training

    ```
    # to get the help about arguments for train.py execution 
    $ python mlmodel/src/train.py -h
    ```

    - execute the first group of experiments:

    ```sh

    $ python mlmodel/src/train.py --execution_name "Variables Sets" --experiments_list "params/Variables Set.yaml"
    
    ```

    - execute the second group of experiments:

    ```sh

    $ python mlmodel/src/train.py --execution_name "Validation Sets" --experiments_list "params/Validation Sets.yaml"
    
    ```

    - start the experiments API for current, this will be used by frontend experiment evaluation application

    ```sh
    $ cd mlmodel/src/
    $ uvicorn experiment_api:app --port 8001 --reload
    
    ```
