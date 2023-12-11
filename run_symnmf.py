import numpy as np
import sys
import symnmf

np.random.seed(0)


def read_data(filename):
    with open(filename, 'r') as file:
        data = [list(map(float, line.strip().split(','))) for line in file]
    return np.array(data)


def initialize_H(W, k):
    m = np.mean(W)
    return np.random.uniform(0, 2 * np.sqrt(m / k), (len(W), k))


def main():
    # Check the number of arguments
    if len(sys.argv) != 4:
        print("Usage: python3 symnmf.py <k> <goal> <file_name>")
        sys.exit(1)

    # Reading command line arguments
    k = int(sys.argv[1])
    goal = sys.argv[2]
    file_name = sys.argv[3]

    # Reading data
    X = read_data(file_name)
    n, d = X.shape  # Get dimensions of X
    X = X.tolist()

    if goal == "symnmf":
        # Calculate normalized similarity matrix using the C extension
        W = symnmf.norm(X, n, d)        
        # Initialize H
        H = initialize_H(W, k)
        H = H.tolist()
        # Get final H using the C extension
        H = symnmf.symnmf(W, H, n, k)
    elif goal == "sym":
        H = symnmf.sym(X, n, d)
    elif goal == "ddg":
        H = symnmf.ddg(X, n, d)
    elif goal == "norm":
        H = symnmf.norm(X, n, d)
    else:
        print("Invalid goal argument!")
        sys.exit(1)

    # Output the required matrix
    for row in H:
        print(','.join(map(lambda x: format(x, '.4f'), row)))


if __name__ == "__main__":
    main()
