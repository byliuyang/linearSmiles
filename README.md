# linearSmile
Smile detectors based on linear regression

## ¯\\\_(ツ)_/¯

The `solve_weight_detector.py` computes the optimal weights `w` and bias term `b` for the linear regression model by deriving the expression for the gradient of the cost function w.r.t. w and b, setting it to 0, and then solving.

The weight is computed as

<div align="center">
	<img width="180" src ="weight.png"/>
</div>

with the cost function of

<div align="center">
	<img width="340" src ="mse.png"/>
</div> 

## Getting Started

### Prerequisite

- Python v3.6.4
- NumPy v1.14.0

### Running

#### Solve Weight classifier
To test out the Solve Weight Classifier, run the following command in the terminal:

```bash
python3 solve_weight_detector.py
```

Here is a sample output:

```bash
Training MSE: 0.052500
Testing MSE: 0.116247

Training Accuracy: 0.895000
Testing Accurary: 0.767505
```

## Authors

- **Ben Hylak** - *Initial work* - [bhylak](https://github.com/bhylak)

- **Yang Liu** - *Initial work* - [byliuyang](https://github.com/byliuyang)

## License
This repo is maintained under MIT license.