import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#yoooooooooooooooo
# Load the Iris dataset
iris_df = pd.read_csv('Iris.csv')

# Check basic statistics
print(iris_df.describe())

# Visualize distribution of each feature
sns.pairplot(iris_df)
plt.show()

# Perform correlation analysis
correlation_matrix = iris_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Investigate relationship between sepal width and petal length
if 'SepalLengthCm' in iris_df.columns:
    sns.jointplot(x='SepalLengthCm', y='PetalLengthCm', data=iris_df)
    plt.show()
else:
    print('The `Sepal_length` variable is not found in the DataFrame.')