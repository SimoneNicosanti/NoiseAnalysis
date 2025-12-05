import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

DATASET_BASE_PATH = "../results/built_dataset"
REGRESSION_BASE_PATH = "../results/regression"


def polynomial_regression(
    model_name: str,
    noise_metric: str,
    dataframe: pd.DataFrame,
    output_names: str,
    train_set_size: int,
    max_regression_degree: int,
) -> None:

    train_set, test_set = dataframe[:train_set_size], dataframe[train_set_size:]

    X_train, Y_train = train_set.drop(output_names, axis=1), train_set[output_names]
    X_test, Y_test = test_set.drop(output_names, axis=1), test_set[output_names]

    fig: plt.Figure
    axes: list[list[plt.Axes]]
    for output_name in output_names:
        curr_Y_train = Y_train[output_name]
        curr_Y_test = Y_test[output_name]

        fig, axes = plt.subplots(max_regression_degree, 2, figsize=(15, 15))
        for degree in range(1, max_regression_degree + 1):
            model = Pipeline(
                [
                    (
                        "poly_features",
                        PolynomialFeatures(
                            degree=degree,
                            include_bias=False,
                            interaction_only=True,
                        ),
                    ),
                    ("lin_reg", LinearRegression()),
                ]
            )

            model.fit(X_train, curr_Y_train)

            print("Degree: ", degree)
            print("\t Train Score >> ", model.score(X_train, curr_Y_train))
            print("\t Test Score >> ", model.score(X_test, curr_Y_test))

            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            axes[degree - 1][0].scatter(curr_Y_train, train_pred, s=10)
            axes[degree - 1][1].scatter(curr_Y_test, test_pred, s=10)

            axes[degree - 1][0].plot(
                [min(curr_Y_train), max(curr_Y_train)],
                [min(curr_Y_train), max(curr_Y_train)],
                color="red",
            )
            axes[degree - 1][1].plot(
                [min(curr_Y_test), max(curr_Y_test)],
                [min(curr_Y_test), max(curr_Y_test)],
                color="red",
            )

            axes[degree - 1][0].vlines(
                curr_Y_train[0],
                ymin=min(curr_Y_train),
                ymax=max(curr_Y_train),
                color="black",
                linestyles="dashed",
            )

            axes[degree - 1][1].vlines(
                curr_Y_train[0],
                ymin=min(curr_Y_train),
                ymax=max(curr_Y_train),
                color="black",
                linestyles="dashed",
            )

            axes[degree - 1][0].set_title(
                f"Degree {degree}. Train Score: {model.score(X_train, curr_Y_train)}"
            )
            axes[degree - 1][1].set_title(
                f"Degree {degree}. Test Score: {model.score(X_test, curr_Y_test)}"
            )

            axes[degree - 1][0].set_xlabel("True Values")
            axes[degree - 1][0].set_ylabel("Predictions")

            axes[degree - 1][1].set_xlabel("True Values")
            axes[degree - 1][1].set_ylabel("Predictions")

        fig.suptitle(
            f"Polynomial Regression. Noise Metric {noise_metric} on Output {output_name} ",
            fontsize=20,
        )
        plt.tight_layout()

        results_path = os.path.join(REGRESSION_BASE_PATH, model_name, noise_metric)
        os.makedirs(results_path, exist_ok=True)
        plt.savefig(f"{results_path}/{output_name}_polynomial.png")

    pass


def regression_tree(
    model_name: str,
    noise_metric: str,
    dataframe: pd.DataFrame,
    output_names: list[str],
    train_set_size: int,
) -> None:

    train_set, test_set = dataframe[:train_set_size], dataframe[train_set_size:]

    X_train, Y_train = train_set.drop(output_names, axis=1), train_set[output_names]
    X_test, Y_test = test_set.drop(output_names, axis=1), test_set[output_names]

    fig: plt.Figure
    axes: list[plt.Axes]
    for output_name in output_names:
        curr_Y_train = Y_train[output_name]
        curr_Y_test = Y_test[output_name]

        tree = GradientBoostingRegressor()
        tree.fit(X_train, curr_Y_train)

        print("Train Score >> ", tree.score(X_train, curr_Y_train))
        print("Test Score >> ", tree.score(X_test, curr_Y_test))

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].scatter(curr_Y_train, tree.predict(X_train), s=10)
        axes[1].scatter(curr_Y_test, tree.predict(X_test), s=10)

        axes[0].plot(
            [min(curr_Y_train), max(curr_Y_train)],
            [min(curr_Y_train), max(curr_Y_train)],
            color="red",
        )
        axes[1].plot(
            [min(curr_Y_test), max(curr_Y_test)],
            [min(curr_Y_test), max(curr_Y_test)],
            color="red",
        )

        axes[0].set_title(f"Train Score: {tree.score(X_train, curr_Y_train)}")
        axes[1].set_title(f"Test Score: {tree.score(X_test, curr_Y_test)}")

        axes[0].set_xlabel("True Values")
        axes[0].set_ylabel("Predictions")

        axes[1].set_xlabel("True Values")
        axes[1].set_ylabel("Predictions")

        fig.suptitle(
            f"Decision Tree Regression. Noise Metric {noise_metric} on Output {output_name} ",
            fontsize=20,
        )
        plt.tight_layout()

        results_path = os.path.join(REGRESSION_BASE_PATH, model_name, noise_metric)
        os.makedirs(results_path, exist_ok=True)
        plt.savefig(f"{results_path}/{output_name}_tree.png")

    pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--family", type=str, required=True, choices=["yolo11"])
    parser.add_argument("--variant", type=str, required=True)

    parser.add_argument(
        "--regression-type", type=str, required=True, choices=["polynomial", "tree"]
    )

    parser.add_argument(
        "--noise-metrics",
        type=str,
        nargs="+",
        required=True,
        choices=[
            "InfNorm",
            "L1Norm",
            "L2Norm",
            "L1NormAvg",
            "L2NormAvg",
            "SignalNoiseRatio",
        ],
    )

    parser.add_argument("--train-set-size", type=int, required=True)
    parser.add_argument("--max-regression-degree", type=int)

    parser.add_argument("--output-names", type=str, nargs="+", required=True)

    args = parser.parse_args()

    if args.regression_type == "polynomial" and not args.max_regression_degree:
        raise ValueError("Please provide a value for max_regression_degree")

    train_set_size = args.train_set_size
    output_names = args.output_names
    model_name = args.family + args.variant

    for noise_metric in args.noise_metrics:
        dataframe_path = f"{DATASET_BASE_PATH}/{model_name}/{noise_metric}.csv"
        if not os.path.exists(dataframe_path):
            raise FileNotFoundError("Dataset not found")
        dataframe = pd.read_csv(dataframe_path)

        if args.regression_type == "polynomial":
            polynomial_regression(
                model_name=model_name,
                noise_metric=noise_metric,
                dataframe=dataframe,
                output_names=output_names,
                train_set_size=train_set_size,
                max_regression_degree=args.max_regression_degree,
            )
        elif args.regression_type == "tree":
            regression_tree(
                model_name=model_name,
                noise_metric=noise_metric,
                dataframe=dataframe,
                output_names=output_names,
                train_set_size=train_set_size,
            )

    pass


if __name__ == "__main__":
    main()
    pass
