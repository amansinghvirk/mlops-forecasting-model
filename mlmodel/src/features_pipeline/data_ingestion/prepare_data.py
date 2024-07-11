"""
module  : prepare_data.py
date    : 05-Nov-2023
author  : Amandeep Singh

description:
    - module contains the logic to create data sets by collecting data from 
      data sources and apply business logics

objects:
    - Datasets:
        object which contains the final datasets used for creating modeling features

"""

import os
import pandas as pd
import numpy as np
import sqlite3


class Datasets:
    """
    Class for preparing and retrieving datasets for model training and evaluation.
    """

    _proj_dir = None
    _db_path = None
    _db_conn = None
    _sql_query = None
    _transactions_df = None
    _stores_df = None
    _holidays_events_df = None
    _oil_df = None
    _stores_trans_df = None

    def __init__(self, proj_dir: str, db_path: str):
        """
        Initializes the Datasets object.

        Args:
            proj_dir (str): The project directory path.
            db_path (str): The path to the database file.
        """
        self._proj_dir = proj_dir
        self._db_path = db_path
        self._prepare_data()

    def _prepare_data(self):
        """
        Prepares the datasets by connecting to the database, retrieving data, applying filters, and merging dataframes.
        """
        self._conn_db()
        self._transactions()
        self._stores()
        self._holidays_events()
        self._oil()
        self._apply_filter()
        self._prepare_stores_trans()
        self._close_conn_db()

    # Rest of the code...


class Datasets:
    _proj_dir = None
    _db_path = None
    _db_conn = None
    _sql_query = None
    _transactions_df = None
    _stores_df = None
    _holidays_events_df = None
    _oil_df = None
    _stores_trans_df = None

    def __init__(self, proj_dir: str, db_path: str):
        self._proj_dir = proj_dir
        self._db_path = db_path
        self._prepare_data()

    def _prepare_data(self):
        self._conn_db()
        self._transactions()
        self._stores()
        self._holidays_events()
        self._oil()
        self._apply_filter()
        self._prepare_stores_trans()
        self._close_conn_db()

    def _get_query(self, query: str):
        try:
            with open(
                os.path.join(self._proj_dir, "queries", query + ".sql"), "r"
            ) as f:
                self._sql_query = f.read()
        except Exception as e:
            self._sql_query = None
            print(e)

    def _get_data(self):
        try:
            df = pd.read_sql_query(self._sql_query, self._db_conn)
            return df
        except Exception as e:
            print(e)

    def _conn_db(self):
        try:
            self._db_conn = sqlite3.connect(self._db_path)
        except Exception as e:
            print(e)

    def _close_conn_db(self):
        try:
            if self._db_conn:
                self._db_conn.close()
        except Exception as e:
            print(e)

    def _transactions(self):
        if self._db_conn:
            self._get_query("transactions")
            self._transactions_df = self._get_data()
            self._transactions_df.date = pd.to_datetime(
                self._transactions_df.date, format="%Y-%m-%d"
            )
        else:
            print("No database connection!")

    def _stores(self):
        if self._db_conn:
            self._get_query("stores")
            self._stores_df = self._get_data()
        else:
            print("No database connection!")

    def _holidays_events(self):
        if self._db_conn:
            # get the data
            self._get_query("holidays_events")
            self._holidays_events_df = self._get_data()
            self._holidays_events_df.date = pd.to_datetime(
                self._holidays_events_df.date, format="%Y-%m-%d"
            )

            # aggregagte
            self._holidays_events_df.loc[
                (
                    (self._holidays_events_df.type == "Event")
                    & (self._holidays_events_df.locale == "National")
                ),
                ["is_national_event"],
            ] = 1
            self._holidays_events_df.loc[
                (
                    (self._holidays_events_df.type == "Holiday")
                    & (self._holidays_events_df.locale == "National")
                ),
                ["is_national_holiday"],
            ] = 1
            self._holidays_events_df.loc[
                (
                    (self._holidays_events_df.type == "Holiday")
                    & (self._holidays_events_df.locale == "Local")
                ),
                ["is_local_holiday"],
            ] = 1
            self._holidays_events_df.loc[
                (
                    (self._holidays_events_df.type == "Holiday")
                    & (self._holidays_events_df.locale == "Regional")
                ),
                ["is_regional_holiday"],
            ] = 1

            self._holidays_events_df = (
                self._holidays_events_df.groupby(["date"])
                .agg(
                    is_national_event=("is_national_event", "sum"),
                    is_national_holiday=("is_national_holiday", "sum"),
                    is_local_holiday=("is_local_holiday", "sum"),
                    is_regional_holiday=("is_local_holiday", "sum"),
                )
                .reset_index()
            )

        else:
            print("No database connection!")

    def _oil(self):
        if self._db_conn:
            self._get_query("oil")
            self._oil_df = self._get_data()
            self._oil_df.date = pd.to_datetime(self._oil_df.date, format="%Y-%m-%d")
            self._oil_df.loc[self._oil_df.dcoilwtico == "", ["dcoilwtico"]] = np.nan
            self._oil_df.dcoilwtico = self._oil_df.dcoilwtico.astype("float")

    def _apply_filter(self):
        self._store_trans_df = self._transactions_df
        self._store_trans_df = pd.merge(
            left=self._store_trans_df,
            right=self._stores_df,
            on=["store_nbr"],
            how="inner",
        )

    def _prepare_stores_trans(self):
        if "sales" in self._store_trans_df.columns:
            self._store_trans_df = (
                self._store_trans_df.groupby(["date"])
                .agg(onpromotion=("onpromotion", "sum"), sales=("sales", "sum"))
                .reset_index()
            )
        else:
            self._store_trans_df = (
                self._store_trans_df.groupby(["date"])
                .agg(onpromotion=("onpromotion", "sum"))
                .reset_index()
            )

        self._store_trans_df = pd.merge(
            left=self._store_trans_df, right=self._oil_df, how="left", on=["date"]
        )
        self._store_trans_df = pd.merge(
            left=self._store_trans_df,
            right=self._holidays_events_df,
            how="left",
            on=["date"],
        )
        self._store_trans_df.is_national_event.fillna(0, inplace=True)
        self._store_trans_df.is_national_holiday.fillna(0, inplace=True)
        self._store_trans_df.is_local_holiday.fillna(0, inplace=True)
        self._store_trans_df.is_regional_holiday.fillna(0, inplace=True)

    def get_transactions(self):
        return self._transactions_df

    def get_stores(self):
        return self._stores_df

    def get_holidays_events(self):
        return self._holidays_events_df

    def get_oil(self):
        return self._oil_df

    def get_stores_transactions(self):
        return self._store_trans_df
