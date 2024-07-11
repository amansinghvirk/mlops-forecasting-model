"""
module  : features.py
date    : 05-Nov-2023
author  : Amandeep Singh

description:
    - module contains the logic to multiple features which will be used in model training
      and inference

objects:
    - Features:
        object which contains the derived features

"""

import random


class Features:
    """
    Class representing the features used in the model.
    """

    _stores_transactions = None
    _stores_transactions_train = None
    _stores_transactions_valid = None

    def __init__(self, datasets):
        """
        Initialize the Features object.

        Parameters:
        - datasets: The datasets object containing the store transactions data.
        """
        self._stores_transactions = datasets.get_stores_transactions()
        # self._create_holiday_features()
        self._create_features()

    def _create_holiday_features(self):
        """
        Create holiday features by filling missing values with 0.
        """
        self._stores_transactions.is_national_event.fillna(0, inplace=True)
        self._stores_transactions.is_national_holiday.fillna(0, inplace=True)
        self._stores_transactions.is_local_holiday.fillna(0, inplace=True)
        self._stores_transactions.is_regional_holiday.fillna(0, inplace=True)

    def _create_features(self):
        """
        Create additional features based on the date column.
        Fill missing values in dcoilwtico column with the mean value.
        """
        self._stores_transactions.loc[
            :, ["day_of_month"]
        ] = self._stores_transactions.date.dt.day
        self._stores_transactions.loc[
            :, ["day_of_week"]
        ] = self._stores_transactions.date.dt.weekday
        self._stores_transactions.loc[
            :, ["month_of_year"]
        ] = self._stores_transactions.date.dt.month
        self._stores_transactions.dcoilwtico.fillna(
            self._stores_transactions.dcoilwtico.mean(), inplace=True
        )

    def prepare_model_data(self, validation_split=0.7, random_seed=123):
        """
        Prepare the model data by selecting the required columns and splitting it into train and validation sets.

        Parameters:
        - validation_split: The proportion of data to be used for validation (default: 0.7).
        - random_seed: The random seed for reproducibility (default: 123).
        """
        self._stores_transactions = self._stores_transactions.loc[
            :,
            [
                "date",
                "dcoilwtico",
                "is_national_event",
                "is_national_holiday",
                "is_local_holiday",
                "is_regional_holiday",
                "day_of_month",
                "day_of_week",
                "month_of_year",
                "onpromotion",
                "sales",
            ],
        ]
        train_count = int(self._stores_transactions.shape[0] * validation_split)
        random.seed(random_seed)
        train_idx = random.sample(
            range(1, self._stores_transactions.shape[0]), train_count
        )
        self._stores_transactions_train = self._stores_transactions.iloc[train_idx, :]
        self._stores_transactions_valid = self._stores_transactions.loc[
            ~self._stores_transactions.index.isin(train_idx), :
        ]

    def prepare_model_data(
        self, train_start_dt, train_end_dt, valid_start_dt, valid_end_dt
    ):
        """
        Prepare the model data by selecting the required columns and splitting it into train and validation sets based on date ranges.

        Parameters:
        - train_start_dt: The start date for the train set.
        - train_end_dt: The end date for the train set.
        - valid_start_dt: The start date for the validation set.
        - valid_end_dt: The end date for the validation set.
        """
        self._stores_transactions = self._stores_transactions.loc[
            :,
            [
                "date",
                "dcoilwtico",
                "is_national_event",
                "is_national_holiday",
                "is_local_holiday",
                "is_regional_holiday",
                "day_of_month",
                "day_of_week",
                "month_of_year",
                "onpromotion",
                "sales",
            ],
        ]

        self._stores_transactions_train = self._stores_transactions.loc[
            (
                (self._stores_transactions.date >= train_start_dt)
                & (self._stores_transactions.date <= train_end_dt)
            ),
            :,
        ]
        self._stores_transactions_valid = self._stores_transactions.loc[
            (
                (self._stores_transactions.date >= valid_start_dt)
                & (self._stores_transactions.date <= valid_end_dt)
            ),
            :,
        ]

    def prepare_inference_data(self, start_dt, end_dt):
        """
        Prepare the inference data by selecting the required columns based on date range.

        Parameters:
        - start_dt: The start date for the inference data.
        - end_dt: The end date for the inference data.
        """
        self._stores_transactions = self._stores_transactions.loc[
            :,
            [
                "date",
                "dcoilwtico",
                "is_national_event",
                "is_national_holiday",
                "is_local_holiday",
                "is_regional_holiday",
                "day_of_month",
                "day_of_week",
                "month_of_year",
                "onpromotion",
            ],
        ]
        print(start_dt, end_dt)
        self._stores_transactions = self._stores_transactions.loc[
            (
                (self._stores_transactions.date >= start_dt)
                & (self._stores_transactions.date <= end_dt)
            ),
            :,
        ]

    def get_stores_trans(self):
        """
        Get the store transactions data.

        Returns:
        - The store transactions data.
        """
        return self._stores_transactions

    def get_model_data(self):
        """
        Get the train and validation sets of the model data.

        Returns:
        - The train and validation sets of the model data.
        """
        return self._stores_transactions_train, self._stores_transactions_valid

    def get_inference_data(self):
        """
        Get the inference data.

        Returns:
        - The inference data.
        """
        return self._stores_transactions
