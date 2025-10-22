"""
Results script: script version of results notebooks. Results stored at "results_v" and "results_v_rf" folder.
"""
import sys, os
from pycfx.datasets import *
from pycfx.models import *
from pycfx.benchmarker import *
from pycfx.helpers.visualisation import *
from pycfx.conformal import *
from pycfx.counterfactual_explanations import *
import argparse

from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run counterfactual benchmarking experiments.")
    parser.add_argument("--dataset", type=str, choices=["CaliforniaHousing", "GermanCredit", "GiveMeSomeCredit", "AdultIncome"], default="CaliforniaHousing", help="Dataset to use (CaliforniaHousing, GermanCredit, GiveMeSomeCredit, AdultIncome)")
    parser.add_argument("--model", type=str, choices=["MLP", "RandomForest"], default="MLP", help="Model to use (RandomForest, MLP)")
    args = parser.parse_args()
    is_rf = args.model.lower() == "randomforest"

    print(f"Start {datetime.now()}")
    mlp_config = {"epochs": 100, "batch_size": 64}
    rf_config = {}

    if args.dataset == "CaliforniaHousing":
        dataset_cls = CaliforniaHousing
    elif args.dataset == "GermanCredit":
        dataset_cls = GermanCreditv2
    elif args.dataset == "GiveMeSomeCredit":
        dataset_cls = GiveMeSomeCredit
        mlp_config = {"epochs": 50, "batch_size": 256}
        rf_config = {"max_n_leaves": 500, "n_estimators": 5}
    elif args.dataset == "AdultIncome":
        dataset_cls = AdultIncome
        mlp_config = {"epochs": 50, "batch_size": 256}
        rf_config = {"max_n_leaves": 500, "n_estimators": 5}
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    dataset = dataset_cls(0.6, 0.2, 0.2)
    model_factories = []

    if is_rf:
        factory = ModelFactory(RandomForestSKLearn, dataset.input_properties, config=rf_config, config_multi={})
        model_factories.append(factory)
    else:
        factory = ModelFactory(PyTorchMLP, dataset.input_properties, config=mlp_config, config_multi={})
        model_factories.append(factory)

    n_factuals_main = 100
    n_repeats = 2
    path = Path("results_v_rf" if is_rf else "results_v")
    use_pretrained = True

    metrics = [
        FailuresMetric(), 
        DistanceMetric(), 
        ValidityMetric(),  
        ImplausibilityMetric(included_prop=0.1), 
        LOFMetric(n_neighbours=20, stratified=True),
        SensitivityMetric(n_sensitivity=25, n_neighbours=4, budget=0.001), 
        StabilityMetric(n_neighbours=8, budget=0.001),
    ]

    conformal_config = {
        "alpha": [0.01, 0.05, 0.1], "scorefn_name": ["linear_logits" if is_rf else "linear2"], "kernel_bandwidth": [0.05, 0.1, 0.15, 0.2], 
    }

    generators = [
        GeneratorFactory([MinDistanceCF], config={}, config_multi={}),

        GeneratorFactory([ConformalCF], config={"conformal_class": SplitConformalPrediction}, config_multi={"conformal_config": {"alpha": [0.01, 0.05, 0.1]}}),

        GeneratorFactory([ConformalCF], config={"conformal_class": ConformalCONFEXTree}, config_multi={
            "conformal_config": conformal_config | {"idx_cat_groups_to_ignore": [[1, 2, 3, 4]]} if args.dataset == "AdultIncome" else conformal_config
        }),
    ]

    if is_rf:
        generators.append(
            GeneratorFactory([FOCUSGenerator], config={"n_iter": 200}, config_multi={}),
            GeneratorFactory([FeatureTweakGenerator], config={"epsilon": 0.01}, config_multi={})
        )
    else:
        f1 = GeneratorFactory([WachterGenerator], config={"mad": True}, config_multi={})
        f2 = GeneratorFactory([SchutGenerator], config={"new": True}, config_multi={})
        f3 = GeneratorFactory([ECCCOGenerator], config={}, config_multi={"conformal_config": {"alpha": [0.01, 0.05, 0.1]}})
        generators.extend([f1, f2, f3])
        

    ## Do not modify below
    print("Initializing CFBenchmarker...")
    benchmarker = CFBenchmarker(dataset, n_factuals_main, n_repeats, metrics, model_factories, generators, path, use_pretrained=use_pretrained)

    print("Setting up models...")
    benchmarker.setup_models()

    print("Evaluating models...")
    benchmarker.evaluate_models()

    print("Setting factuals...")
    benchmarker.set_factuals()

    print("Initializing generators...")
    benchmarker.initialise_generators()

    print("Generating counterfactuals...")
    benchmarker.get_counterfactuals(reset=not use_pretrained)

    print("Evaluating counterfactuals...")
    df_out = benchmarker.evaluate_counterfactuals()

    print("Test conformal...")
    benchmarker.test_conformal()

    print("Generating tables")

    structure = ["MinDist", "Wachter", "Greedy", "ConfexNaive", "ECCCo", "ConfexTree"]
    indent_map = {"ConfexNaive": ["\\alpha"], "ECCCo": ["\\alpha"], "ConfexTree": ["\\alpha", "\\text{bw}"]}
    format_map = {"Wachter": "WachterGenerator", "Greedy": "SchutGenerator", "MinDist": "MinDistanceCF", "ECCCo": "ECCCOGenerator", "ConfexNaive": "SplitConformalPrediction", "ConfexTree": "ConformalCONFEXTree", "\\alpha": "alpha", "\\text{bw}": "kernel_bandwidth"}
    formatted_col_names = ["Distance", "Plausibility", "Implausibility", "Sensitivity $(10^{-1})$", "Stability"]
    benchmarker.generate_table(formatted_col_names, structure, indent_map, format_map, scaling=[(3, 0.1)], dp2=True)


    structure = ["ConfexNaive", "ConfexTree"]
    indent_map = {"ConfexNaive": ["\\alpha"], "ConfexTree": ["\\alpha", "\\text{bw}"]}
    format_map = {"ConfexNaive": "SplitConformalPrediction", "ConfexTree": "ConformalCONFEXTree", "\\alpha": "alpha", "\\text{bw}": "kernel_bandwidth"}
    rc = benchmarker.generate_conformal_table(structure, indent_map, format_map)


    print("Generating figures")
    benchmarker.generate_conformal_sim_plot()
    benchmarker.get_alpha_bandwidth_plots()


    print(f"Evaluation complete. See {path}")
    print(f"End {datetime.now()}")


