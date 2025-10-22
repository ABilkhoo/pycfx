This is the repository that accompanies the paper "CONFEX: Counterfactual Explanations with Conformal Guarantees"

To reproduce the results in the paper, you can run the notebook for each model/dataset pair.

Alternatively, you can run the results_script, an example usage is shown below.
```
nohup python3 ./results_script.py --dataset GermanCredit > german_credit.out 2>&1 &
nohup python3 ./results_script.py --dataset AdultIncome > adult_income.out 2>&1 &
nohup python3 ./results_script.py --dataset CaliforniaHousing > california_housing.out 2>&1 &
nohup python3 ./results_script.py --dataset GiveMeSomeCredit > give_me_some_credit.out 2>&1 &
nohup python3 ./results_script.py --dataset GermanCredit --model RandomForest > german_credit_rf.out 2>&1 &
nohup python3 ./results_script.py --dataset AdultIncome --model RandomForest > adult_income_rf.out 2>&1 &
nohup python3 ./results_script.py --dataset CaliforniaHousing --model RandomForest > california_housing_rf.out  2>&1 &
nohup python3 ./results_script.py --dataset GiveMeSomeCredit --model RandomForest > give_me_some_credit_rf.out 2>&1 &
```
Table 1 and all tables in the appendix can then be retrieved from within the path results_path
To generate Figure 1, step through fig_1.ipynb
To generate Figure 2, step through results_notebook.ipynb