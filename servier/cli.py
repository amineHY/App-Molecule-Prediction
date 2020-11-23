import sys
import click
from servier.src.main import Train, Predict, Evaluate


@click.group()
@click.version_option("1.0.0")
def main():
    """Molecule's basic properties prediction App"""
    pass


@main.command()
@click.option('--data_path', '-d', type=str, required=True, default='servier/data/dataset_single.csv', help="Please enter the path of data in order to train the model")
def train(data_path):
    """Train a machine learning model for prediction and save the pretrained model to disk"""
    click.echo(Train(data_path))


@main.command()
@click.option('--path_x_test', '-p', type=str, required=True, default='servier/data/dataset_single.csv', help="Please enter the path of data in order to perform prediction")
def predict(path_x_test):
    """Perform prediction using a pretrained Machine Learning prediction model"""
    Predict(path_x_test)


@main.command()
@click.option('--y_pred', '-p', type=float, required=True, help='Enter the predicted array')
@click.option('--y_true', '-t', type=float, required=True, help='Enter the true array')
def evaluate(y_pred, y_true):
    """Evaluate the prediction model"""
    click.echo(Evaluate(y_pred, y_true))
