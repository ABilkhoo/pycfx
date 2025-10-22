"""
pycfx/datasets/datasets.py
Base Dataset class and included some included datasets
"""

from pycfx.datasets.input_properties import InputProperties

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_moons, fetch_california_housing
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from math import gamma
from typing import List
from abc import ABC, abstractmethod
from importlib.resources import files

RANDOM_STATE = 2

class Dataset(ABC):
    """
    Base class representing a dataset.
    Note: this class could probably be split into a Dataset class, which defines the dataset and a SplitDataset data class, which provides easy access to a specified train-calib-test split.
    """
    def __init__(self, train_prop: float, calib_prop: float, test_prop: float, **kwargs):
        """
        Initialise a dataset with a specified train-calib-test proportion, and optimal kwargs
        """
        np.random.seed(RANDOM_STATE)
        self.X, self.y, self.input_properties = self.define_dataset(**kwargs)
        self.set_split_indicies(train_prop, test_prop, calib_prop)

    @abstractmethod
    def define_dataset() -> tuple[np.ndarray, np.ndarray, InputProperties]:
        """
        Subclasses override this method to return an array of samples X, labels y and InputProperties instance which describes the dataset
        """
        return None
    
    def get_X_y(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the samples X and labels y of the dataset
        """
        return self.X, self.y
    
    def get_input_properties(self) -> InputProperties:
        """
        Get the InputProperties of the dataset
        """
        return self.input_properties

    def get_name(self) -> str:
        """
        Get the name of the dataset
        """
        return self.__class__.__name__
    
    def get_X_y_split(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the split training, calibration and test data of the dataset:
        Returns (X_train, y_train, X_calib, y_calib, X_test, y_test)
        """
        self.X_train = self.X[self.train_indices]
        self.y_train = self.y[self.train_indices]

        self.X_calib = self.X[self.calib_indices]
        self.y_calib = self.y[self.calib_indices]

        self.X_test = self.X[self.test_indices]
        self.y_test = self.y[self.test_indices]

        return self.X_train, self.y_train, self.X_calib, self.y_calib, self.X_test, self.y_test
    
    def median_pairwise_distances(self, data: np.ndarray) -> np.float_:
        """
        Helper to obtain the median pairwise distance of all elements in the dataset
        """

        pairwise_distances = []
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                pairwise_distances.append(np.linalg.norm(data[i] - data[j]))

        return np.median(pairwise_distances)
    
    def sample_dataset(self, n: int=100, seed: int=1) -> tuple[np.ndarray, np.ndarray]:
        """
        Helper to randomly sample `n` points from the dataset
        """

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.X), size=n, replace=n > len(self.X))
        return self.X[indices], self.y[indices]
    
    def sample_test_dataset(self, n: int=100, seed: int=1) -> tuple[np.ndarray, np.ndarray]:
        """
        Helper to randomly sample `n` points from the dataset's test split
        """

        rng = np.random.default_rng(seed)
        X_test = self.X[self.test_indices]
        y_test = self.y[self.test_indices]

        indices = rng.choice(len(X_test), size=n, replace=n > len(self.X))
        return X_test[indices], y_test[indices]
    
    def compute_radius_from_budget(self, budget: float) -> float:
        """
        Helper to compute a radius of a hypersphere with volume corresponding to a `budget`-proportion of the full feature space.
        Uses InputProperties ranges for features if available, or observed ranges from the dataset if not to obtain the feature space volume. 
        """

        observed_distance = []

        for i, (name, ftype, bound) in enumerate(self.input_properties.get_feature_details()):
            if ftype != "categorical":
                if bound is not None and not np.any(np.isinf(bound)):
                    observed_distance.append(bound[-1] - bound[0])
                else:
                    observed_distance.append(np.max(self.X[:, i]) - np.min(self.X[:, i]))

        d = len(observed_distance)
        V_total = np.prod(observed_distance)
        V_target = budget * V_total

        # Volume of a unit d-ball: pi^(d/2) / Gamma(d/2 + 1)
        unit_ball_vol = np.pi**(d / 2) / gamma(d / 2 + 1)

        # Solve for r: V_target = unit_ball_vol * r^d
        r = (V_target / unit_ball_vol) ** (1 / d)

        return r
    
    def sample_neighbours(self, point: np.ndarray, budget:float =0.05, n_samples: int=3, seed: int=1, use_budget: bool=True) -> np.array:
        """
        Obtain `n_samples` samples of points around `point` with budget `budget` and a random seed `seed`.
        Points are sampled uniformly from a hypersphere with volume corresponding to a `budget`-proportion of the full feature space.
        Set `use_budget` to False to have `budget` directly represent the radius to use.
        """
        rng = np.random.default_rng(seed)
        neighbours = np.repeat([point], n_samples, axis=0)

        if use_budget:
            budget = self.compute_radius_from_budget(budget)

        point_noncat = point[self.input_properties.non_cat_idx]
        d = len(point_noncat)

        directions = rng.normal(size=(n_samples, d))
        directions /= np.linalg.norm(directions, axis=1)[:, None] 

        u = rng.random(n_samples) 
        radii = budget * (u ** (1/d))

        points = point_noncat + directions * radii[:, None]
        neighbours[:, self.input_properties.non_cat_idx] = points


        for i in range(neighbours.shape[0]):
            neighbours[i] = self.input_properties.fix_encoding(neighbours[i])

        return neighbours
    
    def set_split_indicies(self, train_prop: float, test_prop: float, calib_prop: float) -> None:
        """
        Set the train-calibration-test split.
        """

        if train_prop + test_prop + calib_prop <= 0.99:
            raise ValueError("train_size + test_size + calib_size must be equal to 1")

        data_len = len(self.X)
        train_len = int(data_len * train_prop)
        test_len = int(data_len * test_prop)
        calib_len = data_len - train_len - test_len

        self.train_indices = np.arange(train_len)
        self.calib_indices = np.arange(train_len, train_len + calib_len)
        self.test_indices = np.arange(train_len + calib_len, data_len)

    def get_ord_bounds(df, key) -> List[np.float_]:
        """
        Helper to get the possible values a possible ordinal feature `key` can hold, through observation in `df`.
        """
        b = np.arange(df[key].nunique()) / (df[key].nunique() - 1)
        return b.tolist()
    


class SyntheticLinearlySeparable(Dataset):
    """
    Synthetic Linearly Separable (SKLearn)
    """
    
    def define_dataset(self, **kwargs):
        X, y = make_classification(
            n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1, n_samples=500, class_sep=2
        )

        input_properties = InputProperties(feature_names=['x', 'y'],
                                            feature_classes=['numeric', 'numeric'],
                                            bounds=[(-float('inf'), float('inf')), (-float('inf'), float('inf'))],
                                            n_targets=2)

        return X, y, input_properties

class SyntheticMoons(Dataset):
    """
    Synthetic Moons (SKLearn)
    """
    

    def define_dataset(self, **kwargs):
        n = 1000
        X, y = make_moons(n_samples=n, noise=0.3, random_state=RANDOM_STATE)
        
        input_properties = InputProperties(feature_names=['x', 'y'],
                                           feature_classes=['numeric', 'numeric'],
                                           bounds=[(-float('inf'), float('inf')), (-float('inf'), float('inf'))],
                                           n_targets=2)
        
        return X, y, input_properties

    
class SyntheticMulticlass(Dataset):
    """
    Synthetic Multiclass (SKLearn)
    """
     
    def define_dataset(self, **kwargs):
        X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, n_classes=3, random_state=RANDOM_STATE)

        input_properties = InputProperties(feature_names=['x', 'y'],
                                           feature_classes=['numeric', 'numeric'],
                                           bounds=[(-float('inf'), float('inf')), (-float('inf'), float('inf'))],
                                           n_targets=3)

        return X, y, input_properties

class SyntheticBimodal(Dataset):
    """
    Synthetic Bimodal (inspired from Figure 1 in Poyiadzi et al. "FACE: feasible and actionable counterfactual explanations." 2020.)
    Set kwargs 'size' to increase the number of points in the dataset.
    """

    def define_dataset(self, **kwargs):
        size = kwargs.get("size", 3)
        x0_class0 = np.random.normal(0, 0.5, 200*size)
        x1_class0 = np.random.uniform(-0.5, 8, 200*size)  
        X_class0 = np.dstack([x0_class0, x1_class0])[0]

        x0_class1 = np.concatenate([np.random.uniform(-1, 8, 100*size), np.random.normal(4, 0.75, 100*size)])
        x1_class1 = np.concatenate([np.random.normal(0, 0.5, 100*size), np.random.normal(8, 0.6, 100*size)])
        X_class1 = np.dstack([x0_class1, x1_class1])[0]

        X = np.concatenate([X_class0, X_class1])
        y = np.concatenate([[0] * len(x0_class0), [1] * len(x0_class1)])

        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]

        input_properties = InputProperties(feature_names=['x', 'y'],
                                           feature_classes=['numeric', 'numeric'],
                                           bounds=[(-float('inf'), float('inf')), (-float('inf'), float('inf'))],
                                           n_targets=2)
        
        return X, y, input_properties
    

class GermanCredit(Dataset):
    """
    GermanCredit dataset: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.
    Cleaned version obtained via Kaggle https://www.kaggle.com/datasets/uciml/german-credit/data
    Numeric 'Age','Credit amount', 'Duration'  scaled to (0, 1) with MinMax scaler
    Ordinal encoded job, saving account, checking account and categorically encoded Purpose
    """


    def define_dataset(self, **kwargs):
        data_path = files("pycfx.datasets.data") / "german_credit_data.csv"
        df = pd.read_csv(data_path, index_col=0)

        df = df.fillna("no_account")
        df = df.replace({'no_account': 0, 'little': 1, 'moderate': 2, 'rich': 3, 'quite rich': 4, 'male': 0, 'female': 1})

        le = LabelEncoder()
        df["Risk"] = le.fit_transform(df["Risk"])

        y = np.array(df['Risk'])
        del df['Risk']

        X = np.array(df)
        transformer = ColumnTransformer(transformers=[
            ('t1', MinMaxScaler(), [0, 1, 2, 4, 5, 6, 7]),
            ('t2', OneHotEncoder(), [3, 8])],
            remainder='passthrough'
        )
        X = transformer.fit_transform(X)

        input_properties = InputProperties(
                feature_names=['Age', 'Sex', 'Job', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration'] + 
                                [f"housing_{i}" for i in range(3)] + [f"purpose_{i}" for i in range(8)],
                feature_classes=['numeric'] + ['ordinal_normalised'] * 4 + ['numeric'] * 2 + ['categorical'] * 11,
                bounds=[
                    (0, 1), 
                    [0, 1], 
                    Dataset.get_ord_bounds(df, 'Job'),
                    Dataset.get_ord_bounds(df, 'Saving accounts'),
                    Dataset.get_ord_bounds(df, 'Checking account'),
                    (0, 1),
                    (0, 1)
                ] + [None] * 11, #3 for housing, 8 for purpose
                n_targets=2)
        
        return X, y, input_properties
        

class GermanCreditv2(Dataset):
    """
    GermanCredit dataset: Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77.
    Cleaned version obtained via Kaggle https://www.kaggle.com/datasets/uciml/german-credit/data
    Numeric 'Age','Credit amount', 'Duration'  scaled to (0, 1) with MinMax scaler
    Ordinal encoded job, saving account, checking account.
    Variant without categorically encoded Purpose.
    """

    def define_dataset(self, **kwargs):
        data_path = files("pycfx.datasets.data") / "german_credit_data.csv"
        df = pd.read_csv(data_path, index_col=0)

        df = df.fillna("no_account")
        df = df.replace({'no_account': 0, 'little': 1, 'moderate': 2, 'rich': 3, 'quite rich': 4, 'male': 0, 'female': 1})

        le = LabelEncoder()
        df["Risk"] = le.fit_transform(df["Risk"])

        y = np.array(df['Risk'])
        del df['Risk']
        del df['Purpose']

        X = np.array(df)
        transformer = ColumnTransformer(transformers=[
            ('t1', MinMaxScaler(), [0, 1, 2, 4, 5, 6, 7]),
            ('t2', OneHotEncoder(), [3])],
            remainder='passthrough'
        )
        X = transformer.fit_transform(X)

        input_properties = InputProperties(
                feature_names=['Age', 'Sex', 'Job', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration'] + 
                                [f"housing_{i}" for i in range(3)],
                feature_classes=['numeric'] + ['ordinal_normalised'] * 4 + ['numeric'] * 2 + ['categorical'] * 3,
                bounds=[
                    (0, 1), 
                    [0, 1], 
                    Dataset.get_ord_bounds(df, 'Job'),
                    Dataset.get_ord_bounds(df, 'Saving accounts'),
                    Dataset.get_ord_bounds(df, 'Checking account'),
                    (0, 1),
                    (0, 1)
                ] + [None] * 3, # 3 for housing
                n_targets=2)
        
        return X, y, input_properties

class GiveMeSomeCredit(Dataset):
    """
    GiveMeSomeCredit dataset: Credit Fusion and Will Cukierski. Give Me Some Credit. https://kaggle.com/competitions/GiveMeSomeCredit, 2011. Kaggle.
    Obtained via Kaggle https://www.kaggle.com/competitions/GiveMeSomeCredit
    8 Numeric features scaled to (0, 1) with MinMax scaler
    """
      
    def define_dataset(self, **kwargs):
        data_path = files("pycfx.datasets.data.GiveMeSomeCredit") / "cs-training.csv"
        df = pd.read_csv(data_path)
        df = df.fillna(0)

        y  = np.array(df['SeriousDlqin2yrs'])
        del df['SeriousDlqin2yrs']
        del df['Unnamed: 0']
        X = np.array(df)

        transformer = ColumnTransformer(transformers=[
            ('t1', MinMaxScaler(), list(range(df.shape[1]))),
        ],remainder='passthrough')
        X = transformer.fit_transform(X)

        input_properties = InputProperties(
                                           feature_names=['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'],
                                           feature_classes=['numeric'] * 10,
                                           bounds=[(0, 1)] * 10,
                                           n_targets=2)
        
        return X, y, input_properties
    

class CaliforniaHousing(Dataset):
    """
    CaliforniaHousing dataset: Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions, Statistics and Probability Letters, 33 (1997) 291-297
    Obtained from the StatLib repository via SKLearn sklearn.datasets.fetch_california_housing https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
    8 Numeric features scaled to (0, 1) with MinMax scaler
    The regression problem was converted to classification of houses with income above 20000 (42% above, 58% below)
    """
      

    def define_dataset(self, **kwargs):
        california_housing = fetch_california_housing(as_frame=True)

        X = california_housing.data.values
        y = (california_housing.target.values > 2).astype(int)  
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        input_properties = InputProperties(
            feature_names=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'],
            feature_classes=['numeric'] * 8,
            bounds=[(0, 1)] * 8,
            n_targets=2
        )

        return X, y, input_properties


class AdultIncome(Dataset):
    """
    AdultIncome dataset: Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20.
    See https://www.cs.toronto.edu/~delve/data/adult/adultDetail.html
    Obtained via Kaggle https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
    """
      
    def define_dataset(self, **kwargs):
        data_path = files("pycfx.datasets.data") / "adult.csv"
        df = pd.read_csv(data_path)

        del df["education"]
        del df["fnlwgt"]
        del df['native-country']

        df.replace("?", np.nan, inplace=True)
        df.dropna(inplace=True)

        for column in ['income', 'gender']:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])

        y = np.array(df['income'])
        del df['income']

        X = np.array(df)
        transformer = ColumnTransformer(transformers=[
            ('t1', MinMaxScaler(), [0, 2, 7, 8, 9, 10]),
            ('t2', OneHotEncoder(), [1, 3, 4, 5, 6])],
            remainder='passthrough'
        )

        X = np.asarray(transformer.fit_transform(X).todense())

        vals_workclass = pd.unique(df['workclass'])
        vals_marital = pd.unique(df['marital-status'])
        vals_occupation = pd.unique(df['occupation'])
        vals_relationship = pd.unique(df['relationship'])
        vals_race = pd.unique(df['race'])
        n_categorical = len(vals_workclass) + len(vals_marital) + len(vals_occupation) + len(vals_relationship) + len(vals_race)

        input_properties = InputProperties(
            feature_names=['age', 'educational-num', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week'] + 
            [f'workclass_{i}' for i in vals_workclass] + [f'marital_{i}' for i in vals_marital] + [f'occupation_{i}' for i in vals_occupation] + [f'relationship_{i}' for i in vals_relationship] + [f'race_{i}' for i in vals_race],
            feature_classes=['numeric'] + 2 * ['ordinal_normalised']+ 3 * ['numeric'] + n_categorical * ['categorical'],
            bounds=[(0, 1),
                    Dataset.get_ord_bounds(df, 'educational-num'),
                    Dataset.get_ord_bounds(df, 'gender')] + 
                    [(0, 1)] * 3 +
                    [None] * n_categorical,
                    n_targets=2
                )
        
        return X, y, input_properties
            
