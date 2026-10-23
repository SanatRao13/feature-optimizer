# feature-optimizer
Python program to take a given dataset and a model trained on tabular data to reduce dimensionality. Maximizes a certain metric or score while punishing excess feature usage, promotes more efficiency in ML pipelines.


The program uses Bayesian Optimization in order to optimize a hybridized metric. This hybridized metric punishes excess feature inclusion while rewarding improved metrics (i.e. R², the metric that most advanced pipelines struggle with due to overfitting or erratic data). By using this, features aren't given excess attention or weightage, thereby increasing generalizability. As a result, vulnerability to sparse data is also reduced.

This code just simulates retraining the machine learning model across several optimization trials, local test runs thus far have indicated that 100-200 trials is sufficient, especially for highly efficient regressors such as LightGBM, XGBoost, and Random Forest or more primitive ML such as Linear/Lasso/Ridge/Polynomial Regressions. 

**THIS CODE IS NOT MEANT TO BE USED WITH NEURAL NETWORKS OR HIGH COMPUTATIONAL DEMANDS**
The time and space complexity of this algorithm is suited for high-efficiency models with medium size sets (i.e. less than 30 features and less than 10,000 rows/samples), it is not optimized for high-intensity computation that would come from real-world data and would not be effective nor efficient for computationally demanding models such as neural networks or similar. 




## Future developments list
1. Decrease necessity to retrain model from scratch each time -- this requirement increases computational overhead and decreases universality
   - Options:
     - Get a pretrained model and use 0's for eliminated columns?
     - Train the model the first time and force validation predicts with 0'd col.s?
     - Use Lasso/Ridge methods to prioritize features and the use those to minimize retraining required?
2. Expand to Neural Networks
   - MC Dropout already does this to some extent
     - Can be be adapted to achieve the same end result
     - Does this have to be built-in to NN architecture or can be an aux. add-on?
