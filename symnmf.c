#include "symnmf.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


const int max_iter = 300; /** Maximum number of iterations **/
const double epsilon = 1e-4; /** Convergence threshold **/ 

void free_matrix(double** matrix, int n) {
    int i;
    for (i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

double** safe_malloc_matrix(int n, int d) {
    int i, j;
    double** matrix = (double**) malloc(n * sizeof(double*));
    if (!matrix) {
        fprintf(stderr, "Memory allocation failed for matrix rows\n");
        exit(1);
    }

    for (i = 0; i < n; i++) {
        matrix[i] = (double*) malloc(d * sizeof(double));
        if (!matrix[i]) {
            fprintf(stderr, "Memory allocation failed for matrix columns at row %d\n", i);
            
            for (j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            exit(1);
        }
    }

    return matrix;
}

void print_matrix(double** matrix, int n, int m) {
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m - 1; j++) {
            printf("%.4f,", matrix[i][j]);
        }
        printf("%.4f\n", matrix[i][m-1]);
    }
}


double squared_euclidean_distance(const double* a, const double* b, int d) {
    double sum = 0.0;
    int i;
    for (i = 0; i < d; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum;
}

void matrix_multiply(double** A, double** B, double** C, int n, int m, int k) {
    int i, j, p;

    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            C[i][j] = 0;
            for (p = 0; p < m; p++) {
                C[i][j] += A[i][p] * B[p][j];
            }
        }
    }
}

double** matrix_transpose(double** matrix, int rows, int cols) {
    double** transpose = safe_malloc_matrix(cols, rows);
    int i, j;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            transpose[j][i] = matrix[i][j];
        }
    }
    return transpose;
}

double** sym(double** X, int n, int d) {
    double** A = safe_malloc_matrix(n, n);
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j) {
                A[i][j] = exp(-squared_euclidean_distance(X[i], X[j], d) / 2.0);
            } else {
                A[i][j] = 0;
            }
        }
    }
    return A;
}

double** ddg(double** X, int n, int d) {
    double** A = sym(X, n, d);
    double** D = safe_malloc_matrix(n, n);
    double sum;
    int i, j;
    for (i = 0; i < n; i++) {
        sum = 0;
        for (j = 0; j < n; j++) {
            sum += A[i][j];
        }
        D[i][i] = sum;
    }

    free_matrix(A, n);

    return D;
}

double** norm(double** X, int n, int d) {
    double** A = sym(X, n, d);
    double** D = ddg(X, n, d);
    double** W = safe_malloc_matrix(n, n);
    int i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            W[i][j] = A[i][j] / sqrt(D[i][i] * D[j][j]);
        }
    }
    
    free_matrix(A, n);
    free_matrix(D, n);

    return W;
}

double** symnmf(double** W, double** H_initial, int n, int k) {
    const double beta = 0.5;
    int iteration;
    double difference;
    double** H_transpose;
    int i, j;

    double** H = safe_malloc_matrix(n, k);
    double** H_prev = safe_malloc_matrix(n, k);
    double** temp = safe_malloc_matrix(n, k);
    double** temp2 = safe_malloc_matrix(n, n);
    double** temp3 = safe_malloc_matrix(n, k);

    /** Initialize H with H_initial **/ 
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            H[i][j] = H_initial[i][j];
        }
    }

    for (iteration = 0; iteration < max_iter; iteration++) {
        /** Copy current H to H_prev **/
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                H_prev[i][j] = H[i][j];
            }
        }

        /** Compute W * H and store in temp **/
        matrix_multiply(W, H, temp, n, n, k);

        /** Compute H^T and store in a new matrix called H_transpose **/
        H_transpose = matrix_transpose(H, n, k);

        /** Compute H * H^T and store in temp2 **/
        matrix_multiply(H, H_transpose, temp2, n, k, n);
        
        /** Compute temp2 * H and store in temp3 **/
        matrix_multiply(temp2, H, temp3, n, n, k);

        /** Update H **/
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                if (temp3[i][j] != 0.0) {
                    H[i][j] = H[i][j] * (1.0 - beta + beta * temp[i][j] / temp3[i][j]);
                }
            }
        }

        /** Check for convergence **/
        difference = 0.0;
        for (i = 0; i < n; i++) {
            for (j = 0; j < k; j++) {
                difference += (H[i][j] - H_prev[i][j]) * (H[i][j] - H_prev[i][j]);
            }
        }

        if (difference < epsilon) {
            break;
        }
    }

    /** Free memory for temporary matrices **/
    for (i = 0; i < n; i++) {
        free(H_prev[i]);
        free(temp[i]);
        free(temp3[i]);
    }

    for (i = 0; i < k; i++) { 
        free(H_transpose[i]);
    }

    for (i = 0; i < n; i++) {
        free(temp2[i]);
    }

    free(H_prev);
    free(temp);
    free(temp2);
    free(temp3);
    free(H_transpose);

    return H;
}

int main(int argc, char** argv) {
    
    int n; 
    int d; 
    int ch;
    int i,j;
    double** result;
    char* endptr;
    char buffer[100];
    int ret;
    double** X;
    char* goal;
    char* file_name;
    int isBlank;
    FILE* fp;

    if (argc != 3) {
        fprintf(stderr, "Usage: ./symnmf <goal> <data_file>\n");
        exit(1);
    }
    
    goal = argv[1];
    file_name = argv[2];

    /** Read data from file **/
    fp = fopen(file_name, "r");
    if (!fp) {
        fprintf(stderr, "Unable to open the file: %s\n", file_name);
        exit(1);
    }

    isBlank = 1; 
    n = 0;

    while ((ch = fgetc(fp)) != EOF) {
        if (ch != '\n' && ch != '\r' && ch != ' ' && ch != '\t') {
            isBlank = 0; 
        }

        if (ch == '\n') {
            if (!isBlank) {
                n++; 
            }
            isBlank = 1; 
        }
    }

    if (!isBlank) {
        n++;
    }
    rewind(fp);

    d = 0;
    while((ch = fgetc(fp)) != '\n')
    {
        if(ch == ',')
        {
            d++;
        }
    }
    d++;
    rewind(fp);

    X = safe_malloc_matrix(n, d);

    for (i = 0; i < n; i++) {
        for (j = 0; j < d; j++) {
            if (j != d - 1) {
                ret = fscanf(fp, "%s,", buffer);  
            } else {
                ret = fscanf(fp, "%s\n", buffer);  
            }
        
            if (ret == EOF) {
                break;
            }

            if (ret != 1) {  
                if (j != d - 1) {
                    fprintf(stderr, "Failed to read string followed by comma at point %d, dimension %d\n", i, j);
                } else {
                    fprintf(stderr, "Failed to read string followed by newline at point %d, dimension %d\n", i, j);
                }
                free_matrix(X, n);
                exit(1);
            }
        
            X[i][j] = strtod(buffer, &endptr);
            if (endptr == buffer) {
                fprintf(stderr, "Conversion error occurred at point %d, dimension %d\n", i, j);
            
                /** Clean up before exiting **/
                free_matrix(X, n);
                fclose(fp);
                exit(1);
            }
        }
    
        if (ret == EOF) {
            break;
        }
    }
    fclose(fp);


    if (strcmp(goal, "sym") == 0) {
        result = sym(X, n, d);
    } else if (strcmp(goal, "ddg") == 0) {
        result = ddg(X, n, d);
    } else if (strcmp(goal, "norm") == 0) {
        result = norm(X, n, d);
    } else {
        fprintf(stderr, "Invalid goal argument!\n");
        /** Clean up before exiting **/
        free_matrix(X, n);
        exit(1);
    }

    /** Print result **/
    print_matrix(result, n, n);

    /** Cleanup and free memory **/
    for (i = 0; i < n; i++) {
        free(X[i]);
        free(result[i]);
    }
    free(X);
    free(result);

    return 0;
}
