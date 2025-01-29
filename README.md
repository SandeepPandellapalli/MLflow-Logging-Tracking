# MLflow-Logging-Tracking
This repository demonstrates using MLflow for managing and tracking ML experiments. It includes setting up and executing a Logistic Regression model, with detailed tracking of parameters and metrics. Ideal for those interested in MLflow for enhancing model management and reproducibility.

## Problem Statement
In the realm of machine learning, managing and tracking experiments efficiently remains a significant challenge, particularly when dealing with multiple model iterations and configurations. This project addresses the need for a systematic approach to log, compare, and analyze machine learning experiments, using MLflow. The objective is to demonstrate the setup and execution of a logistic regression model, leveraging MLflow's capabilities to provide comprehensive tracking of model parameters, performance metrics, and experiment artifacts, thereby enhancing reproducibility and streamlining the management of machine learning workflows.

## Environment Setup

Create a virtual environment and activate it:

```bash
conda create -p venv python=3.10
conda activate venv/
```

## Install Dependencies

Install the required Python libraries specified in `requirements.txt` (also referred to as `re.txt`):

```bash
pip install -r requirements.txt
```

##  Importing Libraries

```bash
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import mlflow
import mlflow.sklearn
from pathlib import Path
import os
import argparse
```
- mlflow: An open-source platform for managing the end-to-end machine learning lifecycle. It includes tracking experiments to record and compare parameters and results.
- mlflow.sklearn: This module is part of MLflow and is specifically designed to provide utilities to log, load, and serve scikit-learn models using MLflow.
- pathlib.Path: This class in the pathlib module offers classes representing filesystem paths with semantics appropriate for different operating systems. It provides methods to perform tasks related to file paths (like joining paths, reading files, etc.).
- os: Provides a way of using operating system-dependent functionality like reading or writing to a file, manipulating paths, etc.
- argparse: This module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv.
- logging: Provides a flexible framework for emitting log messages from Python programs. It is used to track events that happen when some software runs.
- pandas: An open-source library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language. It is especially good at handling tabular data.
- sklearn.model_selection.train_test_split: Part of the scikit-learn library, this function is used to split datasets into random train and test subsets, which is helpful for evaluating the performance of machine learning models.
- sklearn.linear_model.LogisticRegression: Also from the scikit-learn library, this module is used to perform logistic regression, a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome.
- sklearn.metrics: This module includes score functions, performance metrics, and pairwise metrics and distance computations. It is used to assess the accuracy of models.


