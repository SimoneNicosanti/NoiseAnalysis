import argparse
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

DATASET_BASE_PATH = "../results/built_dataset"
REGRESSION_BASE_PATH = "../results/regression"

MAX_REGRESSION_DEGREE = 4


def draw_fit_results(
    model_family: str,
    model_variant: str,
    noise_metric: str,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    predictor_per_output: dict[str, Pipeline],
    predictor_type: str,
):

    fit_base_dir = (
        f"{REGRESSION_BASE_PATH}/{model_family}/{model_variant}/{predictor_type}/fit"
    )

    os.makedirs(
        fit_base_dir,
        exist_ok=True,
    )

    fig: plt.Figure
    axes: list[list[plt.Axes]]

    num_rows = len(predictor_per_output.keys())
    num_cols = 2

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(15, math.ceil(2.5 * num_rows)),
        sharey="row",
        sharex="row",
    )
    if num_rows == 1:
        axes = [axes]

    for out_idx, out_name in enumerate(predictor_per_output.keys()):
        curr_Y_train = Y_train[out_name]
        curr_Y_test = Y_test[out_name]

        curr_predictor = predictor_per_output[out_name]

        axes[out_idx][0].scatter(curr_Y_train, curr_predictor.predict(X_train), s=10)
        axes[out_idx][1].scatter(curr_Y_test, curr_predictor.predict(X_test), s=10)

        axes[out_idx][0].plot(
            [min(curr_Y_train), max(curr_Y_train)],
            [min(curr_Y_train), max(curr_Y_train)],
            color="red",
        )
        axes[out_idx][1].plot(
            [min(curr_Y_test), max(curr_Y_test)],
            [min(curr_Y_test), max(curr_Y_test)],
            color="red",
        )

        test_score = curr_predictor.score(X_train, curr_Y_train)
        train_score = curr_predictor.score(X_test, curr_Y_test)

        axes[out_idx][0].set_title(f"{out_name} Train. R^2 Score: {test_score:.5f}")
        axes[out_idx][1].set_title(f"{out_name} Test. R^2 Score: {train_score:.5f}")

        axes[out_idx][0].set_xlabel("Real Values")
        axes[out_idx][0].set_ylabel("Predicted Values")

        axes[out_idx][1].set_xlabel("Real Values")
        axes[out_idx][1].set_ylabel("Predicted Values")

        axes[out_idx][0].vlines(
            curr_Y_train[0],
            min(curr_Y_train),
            max(curr_Y_train),
            colors="green",
            label="Not Quantized",
            linestyle="--",
            linewidth=2,
        )
        axes[out_idx][1].vlines(
            curr_Y_train[0],
            min(curr_Y_train),
            max(curr_Y_train),
            colors="green",
            label="Not Quantized",
            linestyle="--",
            linewidth=2,
        )

        axes[out_idx][0].vlines(
            curr_Y_train[1],
            min(curr_Y_train),
            max(curr_Y_train),
            colors="black",
            label="All Quantized",
            linestyle="--",
            linewidth=2,
        )
        axes[out_idx][1].vlines(
            curr_Y_train[1],
            min(curr_Y_train),
            max(curr_Y_train),
            colors="black",
            label="All Quantized",
            linestyle="--",
            linewidth=2,
        )

        axes[out_idx][0].legend()
        axes[out_idx][1].legend()

        residuals = curr_Y_train - curr_predictor.predict(X_train)
        print(residuals.mean())

    fig.suptitle(
        f"{predictor_type} for {model_family} {model_variant} with {noise_metric}"
    )

    plt.tight_layout()

    plt.savefig(
        f"{fit_base_dir}/{noise_metric}.png",
    )
    plt.close()


def write_fitting_params(
    model_family: str,
    model_variant: str,
    metric: str,
    predictor_per_output: dict[str, GridSearchCV],
    predictor_type: str,
):
    params_base_dir = (
        f"{REGRESSION_BASE_PATH}/{model_family}/{model_variant}/{predictor_type}/params"
    )
    os.makedirs(params_base_dir, exist_ok=True)

    predictor_params_names = list(
        list(predictor_per_output.values())[0].best_params_.keys()
    )

    dataframe = pd.DataFrame(
        columns=["output_name"] + predictor_params_names + ["score"]
    )

    for output_name in predictor_per_output.keys():
        curr_predictor = predictor_per_output[output_name]
        dataframe_len = len(dataframe)
        dataframe.loc[dataframe_len] = {
            "output_name": output_name,
            **curr_predictor.best_params_,
            "score": curr_predictor.best_score_,
        }

    dataframe.to_csv(f"{params_base_dir}/{metric}.csv", index=False)

    pass


def polynomial_regression(
    model_family: str,
    model_variant: str,
    noise_metric: str,
    dataframe: pd.DataFrame,
    layers_num: int,
    train_set_size: int,
    eval_folds_num: int,
) -> None:

    train_set, test_set = (
        dataframe[:train_set_size],
        dataframe[train_set_size:],
    )

    output_names = train_set.columns[layers_num:]
    X_train, Y_train = train_set.drop(output_names, axis=1), train_set[output_names]
    X_test, Y_test = test_set.drop(output_names, axis=1), test_set[output_names]

    predictor_per_output = {}

    for output_name in output_names:
        curr_Y_train = Y_train[output_name]

        model_pipe = Pipeline(
            [
                (
                    "poly",
                    PolynomialFeatures(
                        interaction_only=True,
                    ),
                ),
                ("lin", LinearRegression()),
            ]
        )

        param_grid = {
            "poly__degree": list(range(1, MAX_REGRESSION_DEGREE + 1))
            # puoi aggiungere anche "poly__interaction_only": [True, False], ecc.
        }

        gs = GridSearchCV(
            estimator=model_pipe,
            param_grid=param_grid,
            scoring="r2",  # o 'neg_mean_squared_error'
            cv=eval_folds_num,
            n_jobs=-1,
            refit=True,  # rifitta l'intero training sul best param
        )
        fitted_model = gs.fit(X_train, curr_Y_train)
        print(f"{output_name} -> {gs.best_params_}")

        predictor_per_output[output_name] = fitted_model

    draw_fit_results(
        model_family,
        model_variant,
        noise_metric,
        X_train,
        Y_train,
        X_test,
        Y_test,
        predictor_per_output,
        "Polynomial",
    )

    write_fitting_params(
        model_family, model_variant, noise_metric, predictor_per_output, "Polynomial"
    )


def tree_based_regression(
    model_family: str,
    model_variant: str,
    noise_metric: str,
    dataframe: pd.DataFrame,
    layers_num: int,
    train_set_size: int,
    eval_folds_num: int,
) -> None:

    train_set, test_set = (
        dataframe[:train_set_size],
        dataframe[train_set_size:],
    )

    output_names = train_set.columns[layers_num:]
    X_train, Y_train = train_set.drop(output_names, axis=1), train_set[output_names]
    X_test, Y_test = test_set.drop(output_names, axis=1), test_set[output_names]

    predictor_per_output = {}

    for output_name in output_names:
        curr_Y_train = Y_train[output_name]

        model_pipe = Pipeline(
            [
                (
                    "rf",
                    RandomForestRegressor(random_state=42),
                )
            ]
        )

        param_grid = {
            "rf__n_estimators": [50, 100, 200],
            "rf__max_depth": [None, 5, 10, 20],
            "rf__min_samples_split": [2, 5],
            "rf__min_samples_leaf": [1, 2, 4],
        }

        gs = GridSearchCV(
            estimator=model_pipe,
            param_grid=param_grid,
            scoring="r2",
            cv=eval_folds_num,
            n_jobs=-1,
            refit=True,
        )

        fitted_model = gs.fit(X_train, curr_Y_train)
        print(f"{output_name} -> {gs.best_params_}")

        predictor_per_output[output_name] = fitted_model

    draw_fit_results(
        model_family,
        model_variant,
        noise_metric,
        X_train,
        Y_train,
        X_test,
        Y_test,
        predictor_per_output,
        "TreeBased",
    )

    write_fitting_params(
        model_family,
        model_variant,
        noise_metric,
        predictor_per_output,
        "TreeBased",
    )


def gradient_boosted_regression(
    model_family: str,
    model_variant: str,
    noise_metric: str,
    dataframe: pd.DataFrame,
    layers_num: int,
    train_set_size: int,
    eval_folds_num: int,
) -> None:

    train_set, test_set = (
        dataframe[:train_set_size],
        dataframe[train_set_size:],
    )

    output_names = train_set.columns[layers_num:]
    X_train, Y_train = train_set.drop(output_names, axis=1), train_set[output_names]
    X_test, Y_test = test_set.drop(output_names, axis=1), test_set[output_names]

    predictor_per_output = {}

    for output_name in output_names:
        curr_Y_train = Y_train[output_name]

        model_pipe = Pipeline(
            [
                (
                    "gb",
                    GradientBoostingRegressor(random_state=42),
                )
            ]
        )

        param_grid = {
            "gb__n_estimators": [100, 200, 300],
            "gb__learning_rate": [0.01, 0.05, 0.1],
            "gb__max_depth": [3, 5],
            "gb__min_samples_split": [2, 5],
            "gb__min_samples_leaf": [1, 2],
        }

        gs = GridSearchCV(
            estimator=model_pipe,
            param_grid=param_grid,
            scoring="r2",
            cv=eval_folds_num,
            n_jobs=-1,
            refit=True,
        )

        fitted_model = gs.fit(X_train, curr_Y_train)
        print(f"{output_name} -> {gs.best_params_}")

        predictor_per_output[output_name] = fitted_model

    draw_fit_results(
        model_family,
        model_variant,
        noise_metric,
        X_train,
        Y_train,
        X_test,
        Y_test,
        predictor_per_output,
        "GradientBoosted",
    )

    write_fitting_params(
        model_family,
        model_variant,
        noise_metric,
        predictor_per_output,
        "GradientBoosted",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-set-size", type=int, required=True)
    parser.add_argument("--eval-folds-num", type=int, required=True)

    parser.add_argument("--layers-num", type=int, required=True)

    args = parser.parse_args()

    base_path = Path(DATASET_BASE_PATH)

    prediction_funcs = {"Polynomial": polynomial_regression}

    for model_family_info_path in base_path.iterdir():
        model_family = model_family_info_path.name
        for model_variant_info in model_family_info_path.iterdir():
            model_variant = model_variant_info.name

            for noise_metric_info in model_variant_info.iterdir():
                noise_metric = noise_metric_info.name.removesuffix(".csv")

                print("Processing: ", model_family, model_variant, noise_metric)

                dataframe_path = f"{DATASET_BASE_PATH}/{model_family}/{model_variant}/{noise_metric}.csv"
                dataframe = pd.read_csv(dataframe_path)

                predictors_noise_metric_info = {}

                for pred_func_case, pred_func in prediction_funcs.items():
                    predictor_info = pred_func(
                        model_family=model_family,
                        model_variant=model_variant,
                        noise_metric=noise_metric,
                        dataframe=dataframe,
                        layers_num=args.layers_num,
                        train_set_size=args.train_set_size,
                        eval_folds_num=args.eval_folds_num,
                    )
                    predictors_noise_metric_info[pred_func_case] = predictor_info

    pass


if __name__ == "__main__":
    # main()
    polynomial_regression(
        model_family="yolo11",
        model_variant="m-det",
        noise_metric="YoloAccuracyFunction",
        dataframe=pd.read_csv(
            "../results/built_dataset/yolo11/m-det/YoloAccuracyFunction.csv"
        ),
        layers_num=12,
        train_set_size=750,
        eval_folds_num=10,
    )

    tree_based_regression(
        model_family="yolo11",
        model_variant="m-det",
        noise_metric="YoloAccuracyFunction",
        dataframe=pd.read_csv(
            "../results/built_dataset/yolo11/m-det/YoloAccuracyFunction.csv"
        ),
        layers_num=12,
        train_set_size=750,
        eval_folds_num=10,
    )

    gradient_boosted_regression(
        model_family="yolo11",
        model_variant="m-det",
        noise_metric="YoloAccuracyFunction",
        dataframe=pd.read_csv(
            "../results/built_dataset/yolo11/m-det/YoloAccuracyFunction.csv"
        ),
        layers_num=12,
        train_set_size=750,
        eval_folds_num=10,
    )
    pass
