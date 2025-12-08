# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model type: Random Forest Classifier

Framework: scikit-learn

Input features: Numerical + categorical features from the cleaned UCI Census Income dataset, including age, education, occupation, marital status, workclass, capital gain/loss, hours per week, etc.

Target: Binary classification — whether an individual's income is > $50k.

Training split: 80/20 train–test split (or your specific split).

Preprocessing:
One-Hot Encoding for categorical variables
Label Binarizer for the target

## Intended Use
Demonstrate how to build a reproducible ML pipeline, including data processing, model training, evaluation, and slice-based fairness checks.

## Training Data
The data utilized for training this model came from the Census Bureau, and consists of salary information: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
Taken from the same Census dataset via stratified train–test split

Includes all feature columns

Used to evaluate model performance and slice-based metrics such as education, sex, race, etc.

## Metrics
Overall test-set performance:

Precision: 0.7321

Recall: 0.6431

F1 Score: 0.6848

Interpretation:

The model is reasonably good at identifying positive income predictions (precision ~73%).

Recall is lower (~64%), meaning many true positives are missed.

The F1 score (~68%) indicates moderate overall performance for a binary classification task.

Low-performing slices may indicate fairness concerns.

## Ethical Considerations
The dataset contains sensitive demographic attributes, and the predictions may encode historical biases.

Because the model is trained on 1990s census data, societal patterns and workforce demographics have changed.

The model should NOT be used to make real decisions affecting people’s lives.

## Caveats and Recommendations
The dataset is old and not representative of current populations.
Performance varies across demographic slices; mitigation techniques such as re-sampling, fairness constraints, or targeted model tuning may be required.
The model should only be used for learning and demonstration, not production.
