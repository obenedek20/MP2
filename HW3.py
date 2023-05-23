import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

arr = np.array([[0.5,1],[1,0.5],[1,1.5]])
labels = np.array([1,1,-1])

classifier = svm.SVC(kernel="linear")

classifier.fit(arr, labels)

print(classifier.coef_)
print(classifier.intercept_)

xx = np.linspace(0, 2, 100)
yy = np.linspace(0, 2, 100)
X1, X2 = np.meshgrid(xx, yy)
Z = classifier.decision_function(np.c_[X1.ravel(), X2.ravel()])
Z = Z.reshape(X1.shape)

# Plot the data points
plt.scatter(arr[:, 0], arr[:, 1], c=labels, cmap=plt.cm.Paired, edgecolors='k')

# Plot the SVM decision boundary
plt.contour(X1, X2, Z, levels=[0], colors='k')

# Set labels and title
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Decision Boundary')