#ifndef SYM_NMF_H
#define SYM_NMF_H

#include <stdio.h>

/** Function prototypes for the functions you've defined in your .c file **/

double** safe_malloc_matrix(int n, int d);
void print_matrix(double** matrix, int n, int m);
double squared_euclidean_distance(const double* a, const double* b, int d);
void matrix_multiply(double** A, double** B, double** C, int n, int m, int k);
double** sym(double** X, int n, int d);
double** ddg(double** X, int n, int d);
double** norm(double** X, int n, int d);
double** symnmf(double** W, double** H_initial, int n, int k);

/** Global constant declarations, if you choose to put them in the header **/
/** This way, if you ever need these constants elsewhere, you can just include the header **/
extern const int max_iter;
extern const double epsilon;

#endif /** SYM_NMF_H **/
