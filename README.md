# Machine-Learning-Analysis-of-E.-Coli-Promoter-Gene-Sequences
Exploring the interplay between genetics and AI, this project delves into the classification of E. Coli promoter gene sequences. It utilizes a well-curated dataset from the UCI Machine Learning Repository to perform detailed exploratory data analysis, feature selection, and predictive modeling using various advanced machine learning algorithms. The objective is to identify promoter regions with high accuracy, aiding in understanding the genetic mechanisms of E. Coli. The dataset comprises 106 instances of promoter and non-promoter sequences, serving as a basis for evaluating several ML models."

For more details and the dataset, visit the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences

### 1. Logistic Regression
- Accuracy: 0.875
- Precision: 0.842
- Recall: 0.941
- Confusion Matrix: [[12, 3], [1, 16]]

### 2. Random Forest (After Hyperparameter Tuning)
- Best Parameters: {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 50}
- Best Cross-Validation Score: 0.96

### 3. Support Vector Machine (After Hyperparameter Tuning)
- Best Parameters: {'C': 1, 'gamma': 'auto', 'kernel': 'rbf'}
- Best Cross-Validation Score: 0.932

### 4. Neural Network
Test Accuracy: 0.90625 (90.625%)

### Interpretations:

Logistic Regression performed quite well considering its simplicity, with high accuracy, precision, and recall.

Random Forest showed the highest cross-validation score (0.96), indicating it might be the best model among the three. The selected parameters suggest using 50 trees without a set maximum depth.

Support Vector Machine also performed well with a cross-validation score of 0.932, suggesting it's capable of handling the complexity of your data.

Neural Network achieved a test accuracy of approximately 90.625%, which is promising. The accuracy during training and validation remained consistent, suggesting that the model isn't overfitting.

Further Steps:
Model Selection: The Random Forest model seems to have the highest cross-validation score, making it a strong candidate. However, choosing the best model also depends on the specific requirements of your task (e.g., is precision more important than recall?).

Evaluation on External Data: If possible, validate the best-performing model(s) on an external dataset to assess their generalizability.

Performance Tuning: For the Neural Network, you might experiment with different architectures, activation functions, and regularization techniques to see if you can improve performance.

Addressing Warnings: The warnings from TensorFlow indicate deprecated functions and optimization flags. While they don't immediately affect your model's performance, updating to the latest TensorFlow practices is recommended for future-proofing your code.

Domain-Specific Interpretation: Collaborate with domain experts to interpret your models' predictions in the context of bioinformatics and understand their biological implications.
