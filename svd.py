import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread


def svd(X, ranks):
    U, S, VT = np.linalg.svd(X, full_matrices=False)
    S = np.diag(S)
    print(U.shape)
    j = 0
    for rank in ranks:
        X_approx = U[:, :rank] @ S[0:rank, :rank] @ VT[:rank, :]
        plt.figure(j + 1)
        j += 1
        img = plt.imshow(256 - X_approx)
        img.set_cmap('gray')
        plt.axis('off')
        plt.title("r = " + str(rank))
        plt.show()

    plt.figure(1)
    plt.semilogy(np.diag(S))
    plt.title("Singular values")
    plt.show()

    plt.figure(2)
    plt.plot(np.cumsum(np.diag(S)) / np.sum(np.diag(S)))
    plt.title("Singlular values: Cumulative sum")
    plt.show()


def main():
    # plt.rcParams["figure.figsize"] = [16, 8]
    A = imread('images/svd.jpeg')
    X = np.mean(A, -1)
    print(X.shape)
    img = plt.imshow(256 - X)
    img.set_cmap('gray')
    plt.axis('off')
    plt.show()
    ranks = [5, 20, 100]
    svd(X, ranks)


if __name__ == "__main__":
    main()