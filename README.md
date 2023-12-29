# Machine-Learning-Analysis-of-E.-Coli-Promoter-Gene-Sequences
Exploring the interplay between genetics and AI, this project delves into the classification of E. Coli promoter gene sequences. It utilizes a well-curated dataset from the UCI Machine Learning Repository to perform detailed exploratory data analysis, feature selection, and predictive modeling using various advanced machine learning algorithms. The objective is to identify promoter regions with high accuracy, aiding in understanding the genetic mechanisms of E. Coli. The dataset comprises 106 instances of promoter and non-promoter sequences, serving as a basis for evaluating several ML models."

For more details and the dataset, visit the UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/67/molecular+biology+promoter+gene+sequences

## Data Preprocessing
In the data preprocessing phase of this project, we tackled the initial challenge of transforming raw biological sequence data into a form amenable to machine learning algorithms. The dataset, sourced from the UCI Machine Learning Repository, consists of 106 promoter gene sequences from E. coli bacteria. Each sequence was subjected to one-hot encoding—a process that translates the four-letter nucleotide alphabet of A, C, G, and T into a binary matrix. This method is essential as it allows the computational models to interpret and learn from the genetic information. The conversion facilitates the application of various statistical and machine learning techniques to discern patterns indicative of promoter regions within the DNA sequences.

## Exploratory Data Analysis (EDA)

1. Nucleotide Frequency in Sequences: This histogram displays the distribution of each nucleotide within the sequences. The higher frequency of thymine (T) suggests a bias in the dataset towards this nucleotide, which could be indicative of specific biological functions or an artifact of the data collection method. Adenine (A) is also relatively frequent, whereas cytosine (C) and guanine (G) appear less often. This frequency distribution can impact the model's learning if certain nucleotides carry more predictive power.
<img width="413" alt="image" src="https://github.com/ekans24/Machine-Learning-Analysis-of-E.-Coli-Promoter-Gene-Sequences/assets/93953899/1d7b4676-c91d-4bd1-bcfa-f94ab45f7972">

2. Entropy by Position in Sequences: The line graph depicts the entropy at each position across all sequences, providing insight into sequence complexity and conservation. Low entropy points indicate highly conserved positions, which may be critical in determining promoter activity due to potential regulatory function. Conversely, high entropy regions reflect greater variability, possibly accommodating diverse regulatory mechanisms. This plot guides the feature selection process by highlighting positions that contribute most to sequence heterogeneity.
<img width="407" alt="image" src="https://github.com/ekans24/Machine-Learning-Analysis-of-E.-Coli-Promoter-Gene-Sequences/assets/93953899/8c1e0e0d-c66e-4f86-ac26-43922cf6419c">

3. PCA of Encoded Sequences: The scatter plot generated from PCA reveals the clustering pattern of the sequences when reduced to two principal components. The spread of data points suggests variability within the dataset, with some clustering hinting at underlying groups. These clusters could correspond to sequences with similar regulatory functions or genetic relationships. The PCA plot is instrumental in visualizing high-dimensional data and can help in the identification of outliers or data points that warrant further investigation.
<img width="403" alt="image" src="https://github.com/ekans24/Machine-Learning-Analysis-of-E.-Coli-Promoter-Gene-Sequences/assets/93953899/e94f0f53-25b7-4f98-a223-ee4bfc54d24f">

## Feature Selection Methodology
In the feature selection process, we aimed to choose a subset of relevant features from a dataset based on entropy values. Entropy measures the diversity or uncertainty of data at each position within a biological sequence.

### Entropy Thresholds
Two entropy thresholds were defined for this analysis:

- High Entropy Threshold: This threshold was set at the 75th percentile of entropy values. It targeted positions with high diversity, ensuring that we focus on the most variable regions of the sequence.

- Low Entropy Threshold: Conversely, the low entropy threshold was established at the 25th percentile of entropy values. It was used to identify positions with low diversity and higher conservation, allowing us to capture conserved regions.

The choice of these specific percentiles aimed to capture the extremes of entropy values, effectively balancing diversity and conservation in feature selection.

This feature selection process was important because it allowed us to retain essential information while reducing the dimensionality of the data. By selecting features based on entropy thresholds, we could improve model performance, reduce computational complexity, and enhance the interpretability of the results, which are critical aspects of data analysis and machine learning.

## Results

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

The model is constructed using TensorFlow's Keras API, a high-level neural networks library. The architecture of the model is delineated, followed by a discussion of its compilation, training, and evaluation stages.

Model Architecture:

1. Model Type: The architecture utilizes a Sequential model, indicating a linear stack of layers.

2. Input Layer: The first layer is a Dense layer with 128 neurons, receiving input of a size specified by X_train.shape[1]. The layer uses a ReLU activation function, which is effective for non-linear data transformations.

3. Regularization Layers: Two Dropout layers are introduced with a dropout rate of 0.5. These layers are crucial in preventing overfitting by randomly setting a portion of the input units to 0 during training.

4. Intermediate Layer: An additional Dense layer with 64 neurons follows, also using the ReLU activation function. This layer adds depth to the model, allowing for more complex patterns to be learned.

5. Output Layer: The final layer is a Dense layer with a single neuron utilizing a sigmoid activation function, ideal for binary classification problems as it outputs probabilities.


Optimizer: The Adam optimizer is used, known for its efficiency in handling large amounts of data and adaptive learning rate capabilities.
Loss Function: Binary crossentropy is chosen as the loss function, which is standard for binary classification tasks.
Metrics: The model includes accuracy as a metric to monitor during training and evaluation.

For the training process, the model is trained using the fit method with the training data (X_train and y_train). A validation split of 20% is used, allowing for the assessment of model performance on unseen data during training.
The training is set for 50 epochs with a batch size of 10, balancing the speed of training with the model's ability to generalize.

Post-training, the model is evaluated on a separate test dataset (X_test and y_test). The final accuracy of the model on this test set is Test Accuracy: 0.90625 (90.625%), providing an estimate of the model's performance on new, unseen data.

This neural network architecture demonstrates a straightforward yet effective approach to binary classification tasks. The inclusion of Dropout layers to combat overfitting, along with a well-thought-out layer structure, makes this model suitable for a variety of binary classification problems. The use of the Adam optimizer and binary crossentropy loss function further aligns the model's design with best practices in the field of deep learning.

## Interpretations:
- Logistic Regression performed quite well considering its simplicity, with high accuracy, precision, and recall.
- Random Forest showed the highest cross-validation score (0.96), indicating it might be the best model among the three. The selected parameters suggest using 50 trees without a set maximum depth.
- Support Vector Machine also performed well with a cross-validation score of 0.932, suggesting it's capable of handling the complexity of your data.
- Neural Network achieved a test accuracy of approximately 90.625%, which is promising. The accuracy during training and validation remained consistent, suggesting that the model isn't overfitting.

## Further Steps:
- Model Selection: The Random Forest model seems to have the highest cross-validation score, making it a strong candidate. However, choosing the best model also depends on the specific requirements of your task (e.g., is precision more important than recall?).
- Evaluation on External Data: If possible, validate the best-performing model(s) on an external dataset to assess their generalizability.
- Performance Tuning: For the Neural Network, you might experiment with different architectures, activation functions, and regularization techniques to see if you can improve performance.
- Addressing Warnings: The warnings from TensorFlow indicate deprecated functions and optimization flags. While they don't immediately affect your model's performance, updating to the latest TensorFlow practices is recommended for future-proofing your code.
- Domain-Specific Interpretation: Collaborate with domain experts to interpret your models' predictions in the context of bioinformatics and understand their biological implications.

## Conclusion
This project demonstrated the efficacy of machine learning in analyzing and interpreting complex biological data, specifically E. Coli promoter gene sequences. Our models successfully identified key patterns and correlations within the gene sequences, offering insights into their functional aspects. This approach not only enhances our understanding of genetic mechanisms in bacteria but also paves the way for innovative applications in biotechnology and medicine. Future work can extend these findings by integrating more diverse datasets and exploring advanced machine learning techniques, further bridging the gap between computational biology and practical medical applications.
