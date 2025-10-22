"""
pycfx/benchmarker/counterfactual_benchmarker.py
Benchmarker for Counterfactual Explanations
"""

from pycfx.counterfactual_explanations.cf_conformal import ConformalCF
from pycfx.benchmarker.factories import ModelFactory, GeneratorFactory
from pycfx.benchmarker.metrics import CFBenchmarkerMetric, DistanceMetric, ImplausibilityMetric, LOFMetric, SensitivityMetric, StabilityMetric, ValidityMetric, FailuresMetric, define_counterfactual_targets
from pycfx.datasets.datasets import Dataset
from pycfx.conformal.split_conformal import SplitConformalPrediction
from pycfx.conformal.conformal_benchmarker import evaluate_conditional

import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import List, Type
from pathlib import Path
from itertools import product

class CFBenchmarker:
    """
    CFBenchmarker: Benchmarker for Counterfactual Explanations
    
    """
    def __init__(self, 
                 dataset: Dataset, 
                 n_factuals_main: int, 
                 n_repeats: int, 
                 metrics: List[CFBenchmarkerMetric],
                 model_factories: List[ModelFactory], 
                 generator_factories: List[GeneratorFactory], 
                 save_dir: Path="experiments", 
                 use_pretrained: bool=True,
                 id: int=0):
        
        """
        Initialise benchmarker with a `dataset`, number `n_factuals_main` of factuals to include the main bank, number of repeats `n_repeats`,
        list `metrics` of `CFBenchmarkerMetrics` to use, list `model_factory` of `ModelFactory`s to obtain models from,
        list `generator_factories` of `GeneratorFactory`s used to compute CFXs.
        Specify save_dir, use_pretrained and id to save and retrieve results 
        """
        
        self.dataset = dataset
        self.n_factuals_main = n_factuals_main
        self.n_repeats = n_repeats
        self.use_pretrained = use_pretrained
        self.metrics = metrics
        self.model_factories = model_factories
        self.generator_factories = generator_factories

        self.X_train, self.y_train, self.X_calib, self.y_calib, self.X_test, self.y_test = self.dataset.get_X_y_split()
        self.save_dir = save_dir / str(id) / self.dataset.get_name()

        self.models_evaluation_path = self.save_dir / "model_evaluation.json"
        self.factuals_path = self.save_dir / "factuals.json"
        self.counterfactuals_path = self.save_dir / "counterfactuals.json"
        self.eval_path_raw = self.save_dir / "evaluation_raw.json"
        self.eval_path_table = self.save_dir / "evaluation_table.txt"
        self.eval_path_table_2 = self.save_dir / "evaluation_table_2.txt"
        self.figs_save_dir = self.save_dir / "figures"
        self.generators_dir = self.save_dir / "generators"
        self.models_dir = self.save_dir / "models"
        self.conformal_eval_raw_path = self.save_dir / "conformal_eval.json"
        self.conformal_eval_text_path = self.save_dir / "conformal_eval.txt"
        self.conformal_eval_table = self.save_dir / "conformal_table.txt"
        self.additional_conformal = {}

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.figs_save_dir, exist_ok=True)
        os.makedirs(self.generators_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def get_dataset(self) -> Dataset:
        """
        Get the dataset
        """
        return self.dataset

    def setup_models(self) -> None:
        """
        1. Setup all models
        """

        all_models = {}

        for factory in self.model_factories:
            models = factory.train_models(self.X_train, self.y_train, self.n_repeats, self.models_dir, self.use_pretrained)
            
            for model in models:
                all_models[str(model.save_dir)] = model

        self.all_models = all_models
 
    def evaluate_models(self) -> None:
        """
        2. Evaluate all models. Stores evaluation results in save_path / "model_evaluation.json"
        """
        all_evaluations = {}

        for pathname, model in self.all_models.items():
            evaluation = model.load_or_save_evaluation(self.X_test, self.y_test, use_pretrained=self.use_pretrained)
            all_evaluations[pathname] = evaluation
        
        with open(self.models_evaluation_path, 'a') as f:
            json.dump(all_evaluations, f, indent=4)


    def set_factuals(self) -> None:
        """
        3. Set all factuals, the main bank plus any additional banks specified by the metrics. 
        Stores factuals at model_path (within save_dir) / "factuals.json"
        """
        all_factuals = {}

        for pathname, model in self.all_models.items():
            factuals_path = Path(pathname) / "factuals.json"

            if factuals_path.is_file() and self.use_pretrained:
                with open(factuals_path, 'r') as f:
                    factuals = json.load(f)
                    all_factuals[pathname] = factuals

            else:
                seed = model.random_state
                factuals_bank = {}

                x_factuals, y_factuals = self.dataset.sample_dataset(self.n_factuals_main, seed=seed)
                y_target = define_counterfactual_targets(x_factuals, model, self.dataset.input_properties.n_targets)
                
                factuals_bank['main'] = (x_factuals.tolist(), y_target.tolist())

                for metric in self.metrics:
                    cf_bank = metric.get_factuals_bank(model, self.dataset.input_properties, self.dataset, factuals_bank, seed)
                    if cf_bank is not None:
                        key, X_factuals, y_targets = cf_bank
                        factuals_bank[key] = (X_factuals.tolist(), y_targets.tolist())

                all_factuals[pathname] = factuals_bank

                with open(factuals_path, 'w') as f:
                    json.dump(factuals_bank, f, indent=4)
        
        return all_factuals

    def initialise_generators(self) -> None:
        """
        4. Initialise and set up all generators
        """
        self.model_generators = defaultdict(list)

        for pathname, model in self.all_models.items():
            for generator_factory in self.generator_factories:
                generators = generator_factory.setup_generators(model, self.dataset.input_properties, self.X_train, self.y_train, self.X_calib, self.y_calib, self.generators_dir, self.use_pretrained)
                self.model_generators[pathname].extend(generators)

    def get_counterfactuals(self, reset=False) -> None:
        """
        5. Compute counterfactual explanations for all factuals over all generators.
        Stores counterfactuals at model_path (within save_dir) / "counterfactuals.json"
        """

        factuals = self.set_factuals()

        counterfactuals_output = {}

        for model_pathname, generators in self.model_generators.items():
            counterfactuals_path = Path(model_pathname) / "counterfactuals.json"
            factuals_bank = factuals[model_pathname]

            model_counterfactuals = {}
            if counterfactuals_path.is_file():
                with open(counterfactuals_path, 'r') as f:
                    model_counterfactuals = json.load(f)

            for generator in generators:
                print(generator.name())
                counterfactuals_bank = {}

                if model_counterfactuals.get(generator.name()) != None and not reset and self.use_pretrained:
                    print(f"Using saved for {model_pathname}-{generator.name()}")
                    counterfactuals_bank = model_counterfactuals.get(generator.name())
                else:
                    for bank_name, bank_value in factuals_bank.items():
                        bank_factuals, bank_targets = bank_value
                        
                        bank_factuals = np.array(bank_factuals)
                        bank_targets = np.array(bank_targets).astype(int)

                        #TODO add timing
                        counterfactuals = generator.generate_counterfactuals(bank_factuals, bank_targets)
                        counterfactuals_bank[bank_name] = counterfactuals.tolist()
                        model_counterfactuals[generator.name()] = counterfactuals_bank

                        with open(counterfactuals_path, 'w') as f:
                            json.dump(model_counterfactuals, f, indent=4)

                model_counterfactuals[generator.name()] = counterfactuals_bank

            counterfactuals_output[model_pathname] = model_counterfactuals

        return counterfactuals_output

    def evaluate_counterfactuals(self, aggregate_means: bool=False) -> List[pd.DataFrame]:
        """
        6. Evaluate generated counterfactual explanations using the specified metrics. 
        Saves JSON results to save_dir / evaluation_raw.json, and a LaTeX table to save_dir / evaluation_table.txt
        """

        factuals_output = self.set_factuals()
        counterfactuals_output = self.get_counterfactuals()

        model_generator_metrics = {}

        for model_factory in self.model_factories:
            for model_name, model_set in model_factory.get_models_over_repeats().items():
                generator_metrics = defaultdict(list)
                
                for model in model_set:
                    counterfactuals = counterfactuals_output[str(model.save_dir)]
                    factuals_bank = factuals_output[str(model.save_dir)]
                    factuals_bank = {k: (np.array(v[0]), np.array(v[1])) for k, v in factuals_bank.items()}


                    for generator_name, counterfactuals_bank in counterfactuals.items():
                        if generator_metrics.get(generator_name) is None:
                            generator_metrics[generator_name] = defaultdict(list)

                        for metric in self.metrics:
                            metric_results = metric.compute_metric(model, self.dataset.input_properties, self.dataset, factuals_bank, counterfactuals_bank)
                            if isinstance(metric_results, np.ndarray) and not aggregate_means:
                                mean_result = np.nanmean(metric_results)    
                                generator_metrics[generator_name][metric.name()].append(mean_result)
                            else:
                                generator_metrics[generator_name][metric.name()].append(metric_results.tolist())

                combined_metrics = defaultdict(dict)
                for generator, generator_results in generator_metrics.items():
                    for metric, metric_results in generator_results.items():
                        combined_metrics[generator][metric] = {
                            "mean": np.mean(metric_results),
                            "sd": np.std(metric_results)
                        }

                model_generator_metrics[str(model_name)] = {"raw": generator_metrics, "aggregated": combined_metrics}


        with open(self.eval_path_raw, 'w') as f:
            json.dump(model_generator_metrics, f, indent=4)

        print("Writing results to files")

        dfs = []

        with open(self.eval_path_table, 'w') as f:
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for model_type, metrics in model_generator_metrics.items():
                f.write(model_type + "\n")
                df = pd.DataFrame(metrics['aggregated'])
                dfs.append(df)
                df_formatted = df.applymap(lambda x: f"{x['mean']:.3f} ± {x['sd']:.3f}" if isinstance(x, dict) else x)
                z = df_formatted.T
                f.write(z.to_latex())
                f.write('\n\n')
        
        self.model_generator_metrics = model_generator_metrics

        return dfs
    
    def generate_table(self, 
                       formatted_col_names=None, structure: list[str]=[], indent_map: dict[str, list[str]]={}, format_map: dict[str, str]={}, generators_predicate=lambda gen: True,
                       distance_metric=None, plausibility_metric=None, implausibility_metric=None, sensitivity_metric=None, stability_metric=None,
                       validity_metric=None, failures_metric=None, report_failures=True, report_invalidity=True, dp2: bool=False, scaling: list[tuple[int, float]]=[],
                       include_extra=[]) -> List[pd.DataFrame]:
        
        """
        Generate a formatted DataFrame and LaTeX table with this benchmarker's results. Saves to evaluation_table_2.txt.

        Parameters:
        - formatted_col_names: Column names of the table (i.e. metric names)
        - structure: The order in which generators should be presented. 
        - indent_map: config properties that should be displayed indented on the table. Supports up to 2 layers of identation
        - format_map: used to identify raw generator names that map to display names used in structure and indent_map
        - *_metric: specify the metric name for the particular metric. Otherwise will pick the first matching metric
        - report_failures, report_invalidity: if True, will add lines to the text file detailing failures and invalidity
        - dp2: round to 2dp.
        - scaling: specify list of column indices and corresponding scaling factor. E.g. [(3, 0.1)] to scale 3rd column by 10%

        For example usage, see results/results_notebook.ipynb
        """
        
        if not self.eval_path_raw.is_file():
            print("Run evaluate_counterfactuals first!")
            return
        
        with open(self.eval_path_raw) as f:
            model_generator_metrics = json.load(f)


        metrics_objs = {
            "distance_metric": distance_metric or next((m.name() for m in self.metrics if isinstance(m, DistanceMetric)), None),
            "plausibility_metric": plausibility_metric or next((m.name() for m in self.metrics if isinstance(m, LOFMetric)), None),
            "implausibility_metric": implausibility_metric or next((m.name() for m in self.metrics if isinstance(m, ImplausibilityMetric)), None),
            "sensitivity_metric": sensitivity_metric or next((m.name() for m in self.metrics if isinstance(m, SensitivityMetric)), None),
            "stability_metric": stability_metric or next((m.name() for m in self.metrics if isinstance(m, StabilityMetric)), None),
            "failures_metric": failures_metric or next((m.name() for m in self.metrics if isinstance(m, FailuresMetric)), None),
            "validity_metric": validity_metric or next((m.name() for m in self.metrics if isinstance(m, ValidityMetric)), None)
        }

        metrics_cols = [metrics_objs['distance_metric'], metrics_objs['plausibility_metric'], metrics_objs['implausibility_metric'], metrics_objs['sensitivity_metric'], metrics_objs['stability_metric']]
        metrics_cols = [col for col in metrics_cols if col is not None]
        metrics_cols += include_extra

        if formatted_col_names is None or len(formatted_col_names) != len(metrics_cols):
            formatted_col_names = metrics_cols

        full_formatted_columns = ["__generator"] + formatted_col_names

        rows_c = []
        validity_strings = []

        for model_type, metrics in model_generator_metrics.items():
            df = pd.DataFrame(metrics['aggregated']).T
            #Extract validity here
            model_name = Path(model_type).name

            if "Failures" in df and report_failures:
                failures = df['Failures'].map(lambda x: x['mean'])
                failures_str = json.dumps(dict(failures[failures >= 0.01]))
                validity_strings.append(f"{model_name}, Failures " + failures_str)

            if "Validity" in df and report_invalidity:
                validity = df['Validity'].map(lambda x: x['mean'])
                validity_str = json.dumps(dict(validity[validity <= 0.9]))
                validity_strings.append(f"{model_name}, Validity " + validity_str)

            df = df.loc[df.index[df.index.map(generators_predicate)], metrics_cols]
            df.columns = formatted_col_names

            for col_num, scaling_factor in scaling:
                col_name = df.columns[col_num]
                df[col_name] = df[col_name].apply(
                    lambda x: {"mean": x["mean"] * scaling_factor, "sd": x["sd"] * scaling_factor} if isinstance(x, dict) else x
                )
            
            rows_c.append(pd.DataFrame([{"__generator": f"\\textbf{{{model_name}}}"}], columns = full_formatted_columns))
            rows_c.extend(self.format_table_with_structure(df, structure, indent_map, format_map, full_formatted_columns))
                    

        concat = pd.concat(rows_c)
        for element in indent_map.values():
            for v in element:
                if v in concat.columns:
                    del concat[v]

        concat.index = concat["__generator"]
        del concat["__generator"]

        if dp2:
            concat = concat.applymap(lambda x: f"{x['mean']:.2f} ± {x['sd']:.2f}" if isinstance(x, dict) else "")
        else:
            concat = concat.applymap(lambda x: f"{x['mean']:.3f} ± {x['sd']:.3f}" if isinstance(x, dict) else "")

        concat = concat.reset_index()
        concat = concat.rename(columns={"__generator":"Generator"})

        with open(self.eval_path_table_2, "w") as f:
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(concat.to_latex(index=False))
            f.write("\n")
            f.write("\n".join(validity_strings))

        return concat
        
    
    def get_alpha_bandwidth_plots(self, distance_metric: str=None, plausibility_metric: str=None):
        """
        Generate distance or plausibility plots against alpha and kernel bandwidth. Requires that the config of the generators contains 'alpha' and/or 'kernel_bandwidth'.
        Saves to figures directory in save_dir.

        For example usage, see results/results_notebook.ipynb
        """
         
        with open(self.eval_path_raw) as f:
            model_generator_metrics = json.load(f)

        plots = []

        metrics_objs = {
            "Distance": distance_metric or next((m.name() for m in self.metrics if isinstance(m, DistanceMetric)), None),
            "Plausibility": plausibility_metric or next((m.name() for m in self.metrics if isinstance(m, LOFMetric)), None),
        }

        for model_path, metrics in model_generator_metrics.items():
            df =  pd.DataFrame(metrics['aggregated']).T

            df['alpha'] = df.index.to_series().apply(lambda x: re.search(r'"alpha":(\d+\.?\d*)', x).group(1) if re.search(r'"alpha":(\d+\.?\d*)', x) else None)
            df['kernel_bandwidth'] = df.index.to_series().apply(lambda x: re.search(r'"kernel_bandwidth":(\d+\.?\d*)', x).group(1) if re.search(r'"kernel_bandwidth":(\d+\.?\d*)', x) else None)

            for metric, metric_key in metrics_objs.items():
                
                df[f'{metric_key}mean'] = df[metric_key].apply(lambda x: x['mean'] if isinstance(x, dict) else None)
                df[f'{metric_key}sd'] = df[metric_key].apply(lambda x: x['sd'] if isinstance(x, dict) else None)

                df['kernel_bandwidth'] = pd.to_numeric(df['kernel_bandwidth'], errors='coerce')
                df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')

                # Filter rows with non-null kernel_bandwidth and alpha values
                df_filtered = df.dropna(subset=['kernel_bandwidth', 'alpha'])

                # Separate rows with None kernel_bandwidth
                df_none_bandwidth = df[df['kernel_bandwidth'].isnull()]

                # Group by alpha and plot
                plt.figure(figsize=(6, 6))
                alpha_colours = {}

                for alpha, group in df_filtered.groupby('alpha'):
                    group = group.sort_values(by='kernel_bandwidth')  # Ensure the data is sorted by kernel_bandwidth
                    plt.errorbar(
                        group['kernel_bandwidth'], 
                        group[f'{metric_key}mean'], 
                        yerr=group[f'{metric_key}sd'], 
                        fmt='o-', 
                        label=f'alpha={alpha}',
                        capsize=2, elinewidth=1, markeredgewidth=0.5
                    )
                    alpha_colours[alpha] = plt.gca().lines[-1].get_color()  # Store the color of the line

                # Plot horizontal lines for None kernel_bandwidth
                for _, row in df_none_bandwidth.iterrows():
                    if "Split" in row.name:
                        plt.axhline(y=row[f'{metric_key}mean'], linestyle='--', color=alpha_colours.get(row['alpha'], 'gray'))


                plt.xlabel('Kernel Bandwidth')
                plt.ylabel(metric)
                plt.xlim(right=0.2)
                plt.legend()
                plt.grid(True)
                plt.savefig(self.figs_save_dir / f"{Path(model_path).name}--{metric}.png", dpi=300, bbox_inches="tight")
                plots.append(plt.gca())
        
        return plots
    

    def get_means_sds(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Helper function to decompose a DataFrame of dictionary objects containing 'mean' and 'sd' keys into two DataFrames, one for mean and one for sd.
        """

        metrics_agg = pd.DataFrame(df)
        means = metrics_agg.applymap(lambda x: x['mean'] if isinstance(x, dict) else x)
        sds = metrics_agg.applymap(lambda x: x['sd'] if isinstance(x, dict) else x)
        return means.T, sds.T
    
    def set_additional_conformal(self, conformal_classes: List[Type[SplitConformalPrediction]], conformal_config: dict, conformal_config_multi: dict) -> None:
        """
        Specify additional conformal classes and their config/config_multi (similar to GeneratorFactory) for these to be evaluated in .test_conformal().
        """

        for model_path, model in self.all_models.items():
            model_conformals = []

            for vals in product(*conformal_config_multi.values()):
                config = dict(zip(conformal_config_multi.keys(), vals))
                config = conformal_config | config

                for conformal_cls in conformal_classes:
                    conformal = conformal_cls(model, self.dataset.input_properties, config=config, save_path=self.generators_dir, use_pretrained=self.use_pretrained)

                    if conformal.dim_reduction:
                        conformal.dim_reduction.setup(model, self.dataset.input_properties, self.dataset.X_train, self.dataset.y_train, self.generators_dir, self.use_pretrained)

                    conformal.calibrate(self.dataset.X_calib, self.dataset.y_calib)
                    model_conformals.append(conformal)

            self.additional_conformal[model_path] = model_conformals
        

    def test_conformal(self, write_to_file: bool=True) -> dict:
        """
        Evaluate the coverage gap and average set size over four schemes: (1) Marginal (2) Class conditional (3) Random binning (10 bins) (4) Counterfactual Simulation
        Saves to conformal_eval.json and conformal_eval.txt
        """

        model_generators = self.model_generators

        metrics_key = ["Marginal", "Class Conditional", "Random Binning", "Counterfactual Sim"]

        model_results = {}

        if self.conformal_eval_raw_path.is_file() and self.use_pretrained:
            with open(self.conformal_eval_raw_path, 'r') as f:
                model_results = json.load(f)
                if len(model_results) > 0:
                    return model_results

        for model_factory in self.model_factories:
            dfs = {}

            for model_desc, models in model_factory.get_models_over_repeats().items():
                model_metrics_size = {}
                model_metrics_covgap = {}

                for model in models:
                    model_name = str(model.save_dir)
                    
                    generators = model_generators[model_name]
                    conformals = [g.conformal for g in generators if isinstance(g, ConformalCF)]
                    
                    if self.additional_conformal.get(model_name):
                        conformals += self.additional_conformal[model_name]

                    for conformal in tqdm(conformals, desc=f"Evaluating conformals", leave=False):
                        conformal_name = conformal.name()

                        set_size_m, coverage_gap_m, set_size_cc, coverage_gap_cc, set_size_rb, coverage_gap_rb, set_size_cf, coverage_gap_cf = evaluate_conditional(conformal, self.X_test, self.y_test, cov_gap=True)

                        model_metrics_size[(conformal_name, model_name)] = [set_size_m, set_size_cc, set_size_rb, set_size_cf]
                        model_metrics_covgap[(conformal_name, model_name)] = [coverage_gap_m, coverage_gap_cc, coverage_gap_rb, coverage_gap_cf]

                model_dfs = []

                for data_dict in (model_metrics_size, model_metrics_covgap):
                    df = pd.DataFrame(data_dict)
                    df = df.set_index(pd.MultiIndex.from_product([metrics_key], names=["metrics"]))   
                    df_grouped = df.T.groupby(level=0).mean()
                    df_grouped_sd = df.T.groupby(level=0).std()
                    model_dfs.append((df_grouped, df_grouped_sd))

                dfs[str(model_desc)] = {"size": {"mean": model_dfs[0][0], "sd": model_dfs[0][1]}, "covgap": {"mean": model_dfs[1][0], "sd": model_dfs[1][1]}}
            
            model_results |= dfs
        
    
        with open(self.conformal_eval_raw_path, 'w') as f:
            json.dump(model_results, f, indent=4, default=lambda obj: obj.to_json() if isinstance(obj, pd.DataFrame) else None)
                

        if write_to_file:
            with open(self.conformal_eval_text_path, 'a') as f:
                for model_desc, results in dfs.items():
                    mean_df = results["size"]["mean"]
                    sd_df = results["size"]["sd"]
                    formatted_df = mean_df.applymap(lambda x: f"{x:.3f}") + " ± " + sd_df.applymap(lambda x: f"{x:.3f}")

                    mean_df_cg = results["covgap"]["mean"]
                    sd_df_cg = results["covgap"]["sd"]
                    formatted_df_cg = mean_df_cg.applymap(lambda x: f"{x:.3f}") + " ± " + sd_df_cg.applymap(lambda x: f"{x:.3f}")

                    f.write(f"{model_desc}\n")
                    f.write(f"Average set size\n")
                    f.write(formatted_df.to_latex())
                    f.write(f"Coverage gap\n")
                    f.write(formatted_df_cg.to_latex())
                    f.write("\n\n")
        return model_results


    def generate_conformal_table(self, structure: list[str]=[], indent_map: dict[str, list[str]]={}, format_map: dict[str, str]={}, include_size: bool=False):
        """
        Generate a formatted DataFrame and LaTeX table with the conformal evaluatation results. Saves to conformal_table.txt.

        Parameters:
        - structure: The order in which SplitConformalPrediction (+subclasses) instances should be presented. 
        - indent_map: config properties that should be displayed indented on the table. Supports up to 2 layers of identation
        - format_map: used to identify raw generator names that map to display names used in structure and indent_map
        - include_size: add a column for average set size.

        For example usage, see results/results_notebook.ipynb
        """
         
        conformal_results = self.test_conformal()
        
        rows_c = []
        columns = ["__generator", "Marginal CovGap", "Binning CovGap", "Class Cond CovGap", "Simulated CovGap"]
        if include_size:
            columns = ["__generator", "Marginal CovGap", "Margianal SetSize", "Binning Cov", "Class Cond Cov", "Simulated Cov"]

        for model_type, results in conformal_results.items():
            results_size = pd.read_json(results["size"]["mean"])
            results_size_sd = pd.read_json(results["size"]["sd"])

            results_cov = pd.read_json(results["covgap"]["mean"])
            results_cov_sd = pd.read_json(results["covgap"]["sd"])

            size_df = results_size.round(2).map(lambda x: f"{x:.2f}") + " ± " + results_size_sd.round(2).map(lambda x: f"{x:.2f}")
            covgap_df = results_cov.round(2).map(lambda x: f"{x:.2f}") + " ± " + results_cov_sd.round(2).map(lambda x: f"{x:.2f}")

            covgap_df.insert(loc=0, column="__generator", value=covgap_df.index)
            if include_size:
                covgap_df.insert(loc=2, column="Marginal Size", value=size_df["Marginal"])

            covgap_df.columns = columns
            model_name = Path(model_type).name
            
            rows_c.append(pd.DataFrame([{"__generator": f"\\textbf{{{model_name}}}"}], columns = columns))
            rows_c.extend(self.format_table_with_structure(covgap_df, structure, indent_map, format_map, columns))


        concat = pd.concat(rows_c)
        for element in indent_map.values():
            for v in element:
                if v in concat.columns:
                    del concat[v]

        concat.index = concat["__generator"]
        del concat["__generator"]

        concat = concat.reset_index()
        concat = concat.rename(columns={"__generator":"Generator"})
        concat = concat.fillna("")

        with open(self.conformal_eval_table, "w") as f:
            f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(concat.to_latex(index=False))
            f.write("\n")

        return concat

      
    def generate_conformal_sim_plot(self):
        """
        Create a plot of coverage gap against alpha and kernel bandwidth.
        For example usage, see results/results_notebook.ipynb
        """
        conformal_results = self.test_conformal()

        plots = []

        for model_path, results in conformal_results.items():
            cov_gap = pd.read_json(results["covgap"]["mean"])
            cov_gap_errors = pd.read_json(results["covgap"]["sd"])

            cov_gap['alpha'] = cov_gap.index.to_series().apply(lambda x: re.search(r'"alpha":(\d+\.?\d*)', x).group(1) if re.search(r'"alpha":(\d+\.?\d*)', x) else None)
            cov_gap['kernel_bandwidth'] = cov_gap.index.to_series().apply(lambda x: re.search(r'"kernel_bandwidth":(\d+\.?\d*)', x).group(1) if re.search(r'"kernel_bandwidth":(\d+\.?\d*)', x) else None)

            cov_gap_errors['alpha'] = cov_gap_errors.index.to_series().apply(lambda x: re.search(r'"alpha":(\d+\.?\d*)', x).group(1) if re.search(r'"alpha":(\d+\.?\d*)', x) else None)
            cov_gap_errors['kernel_bandwidth'] = cov_gap_errors.index.to_series().apply(lambda x: re.search(r'"kernel_bandwidth":(\d+\.?\d*)', x).group(1) if re.search(r'"kernel_bandwidth":(\d+\.?\d*)', x) else None)


            df_filtered = cov_gap.dropna(subset=['alpha'])

            df_filtered['kernel_bandwidth'] = pd.to_numeric(df_filtered['kernel_bandwidth'], errors='coerce')
            df_filtered['Counterfactual Sim'] = pd.to_numeric(df_filtered['Counterfactual Sim'], errors='coerce')

            df_none_bandwidth = df_filtered[df_filtered['kernel_bandwidth'].isnull()]
            df_filtered = df_filtered.dropna(subset=['kernel_bandwidth'])

            alpha_colours = {}

            plt.figure(figsize=(6, 6))
            for alpha, group in df_filtered.groupby('alpha'):
                group = group.sort_values(by='kernel_bandwidth')  # Ensure the data is sorted by kernel_bandwidth
                plt.errorbar(
                    group['kernel_bandwidth'], 
                    group['Counterfactual Sim'], 
                    yerr=cov_gap_errors.loc[group.index, 'Counterfactual Sim'], 
                    fmt='o-', 
                    label=f'alpha={alpha}',
                    capsize=2, elinewidth=0.7, markeredgewidth=1
                )
                alpha_colours[alpha] = plt.gca().lines[-1].get_color()  # Store the color of the line

            # Plot horizontal lines for None kernel_bandwidth
            for _, row in df_none_bandwidth.iterrows():
                plt.axhline(y=row['Counterfactual Sim'], linestyle='--', c=alpha_colours[row['alpha']])

            plt.axhline(y=0, color='red', linestyle='--', label='Target')

            # plt.title('Coverage gap for Simulated CFXs')
            plt.xlabel('Kernel Bandwidth')
            plt.ylabel('Coverage gap')
            plt.legend()
            plt.xlim(right=0.2)
            plt.grid(True)
            plt.savefig(self.figs_save_dir / f"{Path(model_path).name}--conformalsim.png", dpi=300, bbox_inches='tight')
            plt.show()

        return plots


    def format_table_with_structure(self, df: pd.DataFrame, structure: list[str], indent_map: dict[str, list[str]], format_map: dict[str, str], columns: list[str]):
        """
        Helper to generate DataFrame rows to be concatenated into a formatted table following structure, indent_map and format_map

        Parameters:
        - structure: The order in which SplitConformalPrediction (+subclasses) instances should be presented. 
        - indent_map: config properties that should be displayed indented on the table. Supports up to 2 layers of identation
        - format_map: used to identify raw generator names that map to display names used in structure and indent_map
        - columns: names of columns in df

        """
        rows_c = []

        for label in structure:
            df_label = format_map.get(label, label)
            if df_label is None:
                continue

            rows = df[df.index.str.contains(df_label)]

            if len(rows) == 0:
                continue

            indent_config = indent_map.get(label)

            if indent_config is None:
                rows.loc[:, "__generator"] = label
                rows_c.append(rows)
            elif indent_config == "config":
                rows_c.append(rows)

            elif len(indent_config) == 1:
                indent_config = indent_config[0]
                sep_row = pd.DataFrame([{"__generator": label}], columns = columns)

                formatted_index_config = format_map.get(indent_config, indent_config)
                
                regex = f'"{formatted_index_config}"' + r':(\d+\.?\d*)'
                rows.loc[:, indent_config] = rows.index.to_series().apply(lambda x: re.search(regex, x).group(1) if re.search(regex, x) else None)

                rows = rows.sort_values(by=indent_config)

                rows.loc[:, "__generator"] = f'$\\ \\ {indent_config} = ' + rows[indent_config].astype(str) + "$"
                
                rows_c.append(sep_row)
                rows_c.append(rows)
            elif len(indent_config) == 2:
                indent_config_formatted = list(map(lambda x: format_map.get(x, x), indent_config))

                for i, element in enumerate(indent_config):
                    element_formatted = indent_config_formatted[i]
                    regex = f'"{element_formatted}"' + r':(\d+\.?\d*)'
                    rows.loc[:, element] = rows.index.to_series().apply(lambda x: re.search(regex, x).group(1) if re.search(regex, x) else None)

                for indent_val, indent_rows in rows.groupby(indent_config[0], sort=True):
                    sep_row = pd.DataFrame([{"__generator": f"{label}, ${indent_config[0]} = {indent_val}$"}], columns = columns)

                    indent_rows = indent_rows.sort_values(by=indent_config[1])
                        
                    indent_rows.loc[:, "__generator"] = f'$\\ \\ {indent_config[1]} = ' + indent_rows[indent_config[1]].astype(str) + "$"

                    rows_c.append(sep_row)
                    rows_c.append(indent_rows)

            else:
                raise NotImplementedError("Max supported index depth is 2")
            
        return rows_c


