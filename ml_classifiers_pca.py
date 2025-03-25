import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


with open('/Users/chenab/Downloads/train-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    print(data.shape)
    Xtraindata = np.transpose(data.reshape((size, nrows*ncols)))

with open('/Users/chenab/Downloads/train-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    ytrainlabels = data.reshape((size,)) # (Optional)

with open('/Users/chenab/Downloads/t10k-images.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    Xtestdata = np.transpose(data.reshape((size, nrows*ncols)))

with open('/Users/chenab/Downloads/t10k-labels.idx1-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    ytestlabels = data.reshape((size,)) # (Optional)
        

    
traindata_imgs =  np.transpose(Xtraindata).reshape((60000,28,28))    
print(Xtraindata.shape)
print(ytrainlabels.shape)
print(Xtestdata.shape)
print(ytestlabels.shape)

def plot_digits(XX, N, title):
    fig, ax = plt.subplots(N, N, figsize=(8, 8))
    
    for i in range(N):
      for j in range(N):
        ax[i,j].imshow(XX[:,(N)*i+j].reshape((28, 28)), cmap="Greys")
        ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)
    plt.show()

#plot_digits(Xtraindata, 8, "First 64 Training Images" )


X_train = np.transpose(Xtraindata)
X_test = np.transpose(Xtestdata)

print(X_train.shape)
print(X_test.shape)

X_mean = np.mean(X_train, axis=0)
X_train_centered = X_train - X_mean


pca_full = PCA()
pca_full.fit(X_train_centered)

plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(pca_full.components_[i].reshape((28, 28)), cmap="Greys")
    plt.axis("off")
plt.suptitle("First 16 PCA Components", fontsize=24)
plt.show()

#TASK 2
explained_variance_ratio = pca_full.explained_variance_ratio_
cum_explained_variance = np.cumsum(explained_variance_ratio)
k = np.argmax(cum_explained_variance >= 0.85) + 1
print(f"Number of principal components needed to explain 85% of the variance: {k}")  

plt.figure(figsize=(10, 6))
plt.title("Cumulative Explained Variance with 85% threshold", fontsize=16)
plt.xlabel("Number of Principal Components", fontsize=14)
plt.ylabel("Cumulative Explained Variance", fontsize=14)
plt.plot(cum_explained_variance)
plt.axvline(x=k, color='red', linestyle='--', label='k = ' + str(k))
plt.legend()
plt.show()


pca = PCA(n_components=k)
X_train_pca = pca.fit_transform(X_train_centered)
X_test_pca = pca.transform(X_test - X_mean)

X_reconstructed = pca.inverse_transform(X_train_pca)
X_reconstructed_T = X_reconstructed.T

plot_digits(Xtraindata, 8, "First 64 Training Images" )
plot_digits(X_reconstructed_T, 8, "First 64 Reconstructed Images" )

#TASK 3
def select_subset_digits(X_train_pca, ytrain, X_test_pca, y_test, digits):
    train_mask = np.isin(ytrain, digits)
    test_mask = np.isin(y_test, digits)
    return X_train_pca[train_mask], ytrain[train_mask], X_test_pca[test_mask], y_test[test_mask]


X_train_subset_18, y_train_subset_18, X_test_subset_18, y_test_subset_18 = select_subset_digits(X_train_pca, ytrainlabels, X_test_pca, ytestlabels, [1,8])
X_train_subset_38, y_train_subset_38, X_test_subset_38, y_test_subset_38 = select_subset_digits(X_train_pca, ytrainlabels, X_test_pca, ytestlabels, [3,8])
X_train_subset_27, y_train_subset_27, X_test_subset_27, y_test_subset_27 = select_subset_digits(X_train_pca, ytrainlabels, X_test_pca, ytestlabels, [2,7])

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import cross_val_score

ridgeCL = RidgeClassifierCV()
cv_score_18 = cross_val_score(ridgeCL, X_train_subset_18, y_train_subset_18, cv=5)
cv_score_38 = cross_val_score(ridgeCL, X_train_subset_38, y_train_subset_38, cv=5)
cv_score_27 = cross_val_score(ridgeCL, X_train_subset_27, y_train_subset_27, cv=5)

print("{:.2f} accuracy for [1, 8] with a standard deviation of {:.2f}".format(cv_score_18.mean(), cv_score_18.std()))
print("{:.2f} accuracy for [3, 8] with a standard deviation of {:.2f}".format(cv_score_38.mean(), cv_score_38.std()))
print("{:.2f} accuracy for [2, 7] with a standard deviation of {:.2f}".format(cv_score_27.mean(), cv_score_27.std()))

powers = np.arange(-10, 11, 1, dtype=float)
alphas = 10**powers

from sklearn.metrics import accuracy_score

def get_accuracies(X_train_subset, y_train_subset, X_test_subset, y_test_subset, alphas):
    accuracies = []
    for alpha in alphas:
        ridge_classifier = RidgeClassifier(alpha=alpha)
        ridge_classifier.fit(X_train_subset, y_train_subset)
        y_pred = ridge_classifier.predict(X_test_subset)
        accuracy = accuracy_score(y_test_subset, y_pred)
        accuracies.append(accuracy)
    return accuracies

def get_best_alpha(accuracies):
    return alphas[np.argmax(accuracies)]


accuracies_18 = get_accuracies(X_train_subset_18, y_train_subset_18, X_test_subset_18, y_test_subset_18, alphas)
accuracies_38 = get_accuracies(X_train_subset_38, y_train_subset_38, X_test_subset_38, y_test_subset_38, alphas)
accuracies_27 = get_accuracies(X_train_subset_27, y_train_subset_27, X_test_subset_27, y_test_subset_27, alphas)


def plot_accuracies(accuracies, title):
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=16)
    plt.xlabel("Alpha", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.xscale("log")
    best_alpha = get_best_alpha(accuracies)
    plt.axvline(x=best_alpha, color='red', linestyle='--', label='Best Alpha: ' + str(best_alpha))
    plt.axhline(y=accuracies[np.argmax(accuracies)], color='green', linestyle='--', label='Best Accuracy: ' + f"{accuracies[np.argmax(accuracies)]*100:.2f}%")
    plt.legend(fontsize=14)
    plt.plot(alphas, accuracies)
    plt.show()

plot_accuracies(accuracies_18, "Accuracy of Ridge Classifier on Test Set with different alpha values for [1,8]")
plot_accuracies(accuracies_38, "Accuracy of Ridge Classifier on Test Set with different alpha values for [3,8]")
plot_accuracies(accuracies_27, "Accuracy of Ridge Classifier on Test Set with different alpha values for [2,7]")



from sklearn.model_selection import GridSearchCV
powers = np.arange(-20, 21, 1, dtype=float)
alphas = 10**powers

ridge_classifier = RidgeClassifierCV()


parameters = {'alphas': [alphas]} 

grid_search = GridSearchCV(ridge_classifier, parameters, cv=5)
grid_search.fit(X_train_pca, ytrainlabels)

best_alpha = grid_search.best_params_['alphas'][0]

print(f"best alpha for ridge classifier: {best_alpha}")

ridge_classifier = RidgeClassifier(alpha=best_alpha)
ridge_classifier.fit(X_train_pca, ytrainlabels)

ridge_pred = ridge_classifier.predict(X_test_pca)

ridge_accuracy = accuracy_score(ytestlabels, ridge_pred)
print(f"accuracy of ridge classifier: {ridge_accuracy*100:.2f}%")

cv_score_ridge = cross_val_score(ridge_classifier, X_train_pca, ytrainlabels, cv=5)
print(f"cross validation score of ridge classifier: {cv_score_ridge.mean()*100:.2f}%")

#KNN
from sklearn.neighbors import KNeighborsClassifier

k_values = np.arange(1, 20, 2)

accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_pca, ytrainlabels)
    y_pred = knn.predict(X_test_pca)
    accuracy = accuracy_score(ytestlabels, y_pred)
    accuracies.append(accuracy)

best_k = k_values[np.argmax(accuracies)]
print(f"best k for KNN: {best_k}")
print(f"accuracy of KNN: {accuracies[np.argmax(accuracies)]*100:.2f}%")

plt.figure(figsize=(10, 6))
plt.title("Accuracy of KNN on Test Set with different k values", fontsize=16)
plt.xlabel("k", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.axvline(x=best_k, color='red', linestyle='--', label='Best k: ' + str(best_k))
plt.axhline(y=accuracies[np.argmax(accuracies)], color='green', linestyle='--', label='Best Accuracy: ' + f"{accuracies[np.argmax(accuracies)]*100:.2f}%")
plt.legend(fontsize=14)
plt.plot(k_values, accuracies)
plt.show()

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_pca, ytrainlabels)
bestknn_pred = best_knn.predict(X_test_pca)
knn_accuracy = accuracy_score(ytestlabels, bestknn_pred)
print(f"accuracy of KNN with best k: {knn_accuracy*100:.2f}%")




#Confusion Matrices

from sklearn.metrics import confusion_matrix
import seaborn as sns
# Ridge Classifier
plt.figure(figsize=(10, 8))
cm_ridge = confusion_matrix(ytestlabels, ridge_pred)
sns.heatmap(cm_ridge, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for Ridge Classifier", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.show()

# KNN
plt.figure(figsize=(10, 8))
cm_knn = confusion_matrix(ytestlabels, bestknn_pred)
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for KNN with best k", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)

plt.show()

from sklearn.metrics import classification_report

print("Ridge Classifier Classification Report:")
print(classification_report(ytestlabels, ridge_pred))

print("KNN Classification Report:")
print(classification_report(ytestlabels, bestknn_pred))

#QDA
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


param_grid = {'reg_param': np.linspace(0, 1, 11)}

qda_classifier = QuadraticDiscriminantAnalysis()

grid_search_qda = GridSearchCV(qda_classifier, param_grid, cv=5)
grid_search_qda.fit(X_train_pca, ytrainlabels)

best_reg_param = grid_search_qda.best_params_['reg_param']
print(f"Best reg_param for QDA: {best_reg_param}")

qda_best = QuadraticDiscriminantAnalysis(reg_param=best_reg_param)
qda_best.fit(X_train_pca, ytrainlabels)

qda_pred = qda_best.predict(X_test_pca)

qda_accuracy = accuracy_score(ytestlabels, qda_pred)
print(f"QDA Accuracy with best reg_param: {qda_accuracy*100:.2f}%")

cv_score_qda_best = cross_val_score(qda_best, X_train_pca, ytrainlabels, cv=5)
print(f"Cross Validation Score for QDA with best reg_param: {cv_score_qda_best.mean()*100:.2f}%")

plt.figure(figsize=(10, 8))
cm_qda_best = confusion_matrix(ytestlabels, qda_pred)
sns.heatmap(cm_qda_best, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix for QDA (Best reg_param)", fontsize=16)
plt.xlabel("Predicted Label", fontsize=14)
plt.ylabel("True Label", fontsize=14)
plt.show()

print("Quadratic Discriminant Analysis (GridSearchCV) Classification Report:")
print(classification_report(ytestlabels, qda_pred))