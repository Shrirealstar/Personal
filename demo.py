#1
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('tips')
num_col, cat_col = 'total_bill', 'sex'

print(df[num_col].describe())

sns.histplot(df[num_col], kde=True); plt.show()
sns.boxplot(x=df[num_col]); plt.show()

q1, q3 = df[num_col].quantile([0.25, 0.75])
iqr = q3 - q1
outliers = df[(df[num_col] < q1 - 1.5 * iqr) | (df[num_col] > q3 + 1.5 * iqr)]
print("Outliers:\n", outliers)

df[cat_col].value_counts().plot(kind='bar'); plt.show()
df[cat_col].value_counts().plot(kind='pie', autopct='%1.1f%%'); plt.show()


#2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('iris')
n1, n2 = 'sepal_length', 'sepal_width'

sns.scatterplot(x=df[n1], y=df[n2]); plt.show()
print(f'Pearson Correlation: {df[n1].corr(df[n2])}')

corr_matrix = df.select_dtypes(include='number').corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm'); plt.show()

#3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = StandardScaler().fit_transform(iris.data), iris.target

cov_matrix = np.cov(X.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
X_pca = X @ eigenvectors[:, idx[:2]]

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.colorbar(label='Species')
plt.show()

print("Explained variance ratio:", eigenvalues[idx[:2]] / eigenvalues.sum())

#4
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.linspace(0, 10, 100)[:, None]
y = np.sin(X).ravel() + np.random.normal(scale=0.2, size=X.shape[0])

def lwr(x, y, xq, tau):
    W = np.exp(-((x - xq) ** 2) / (2 * tau ** 2))
    W = np.diag(W.ravel())
    X_aug = np.c_[np.ones_like(x), x]
    theta = np.linalg.pinv(X_aug.T @ W @ X_aug) @ (X_aug.T @ W @ y)
    return np.array([1, xq]) @ theta

y_pred = np.array([lwr(X, y, xq, 0.5) for xq in X.ravel()])

plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X, y_pred, color='red')
plt.show()
