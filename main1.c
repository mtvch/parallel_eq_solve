#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct Matrix Matrix;

struct Matrix {
    double *elements;
    size_t row_size;
    size_t col_size;
};

const int max_element_value = 20;
const int overload_value = 50;
const double thao = 0.001;
const double epsilon = 0.00001;

double get_element(const Matrix *A, size_t i, size_t j) {
    if (i >= A->col_size || j >= A->row_size) {
        printf("Out of range");
        return -1;
    }
    return A->elements[i * A->row_size + j];
}

void set_element(const Matrix *A, size_t i, size_t j, double value) {
    A->elements[i * A->row_size + j] = value;
}

void free_matrix(Matrix *A) {
    free(A->elements);
    free(A);
}

Matrix* create_uninit_matrix(size_t col_size, size_t row_size) {
    Matrix *A = malloc(sizeof(Matrix));
    if (A == NULL) {
        printf("Memory error\n");
        return NULL;
    }
    A->elements = malloc(row_size * col_size * sizeof(double));
    if (A->elements == NULL) {
        printf("Memory error\n");
        return NULL;
    }
    A->col_size = col_size;
    A->row_size = row_size;
    return A;
}

size_t power(size_t a, size_t b) {
    if (b == 0) {
        return 1;
    }
    else {
        return a * power(a, b - 1);
    }
}

double hash(size_t i, size_t j, int max_element_value) {
    return (power(i + j, i + j) % max_element_value) * ((double) (i + 1) / (i + j + 2));
}

void fill_random_matrix(Matrix *A) {
    for (size_t i = 0; i < A->col_size; i++) {
        for (size_t j = 0; j < A->row_size; j++) {
            double rand_value = hash(i, j, max_element_value);
            set_element(A, i, j, rand_value);
        }
    }
}

Matrix* create_random_matrix(size_t col_size, size_t row_size) {
    Matrix *A = create_uninit_matrix(col_size, row_size);
    if (A == NULL) {
        return NULL;
    }
    
    fill_random_matrix(A);
    return A;
}

void fill_random_sym_matrix(Matrix *A) {
    for (size_t i = 0; i < A->col_size; i++) {
        for (size_t j = i; j < A->row_size; j++) {
            double rand_value = hash(i, j, max_element_value);
            if (i == j) {
                rand_value += overload_value;
            }
            set_element(A, i, j, rand_value);
        }
    }

    for (size_t i = 1; i < A->col_size; i++) {
        for (size_t j = 0; j < i; j++) {
            set_element(A, i, j, get_element(A, j, i));
        }
    }
}

Matrix* create_zero_matrix(size_t col_size, size_t row_size) {
    Matrix *A = create_uninit_matrix(col_size, row_size);
    if (A == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < col_size; i++) {
        for (size_t j = 0; j < row_size; j++) {
            set_element(A, i, j, 0);
        }
    }
    return A;
}

void print_matrix(Matrix *A) {
    for (size_t i = 0; i < A->col_size; i++) {
        for (size_t j = 0; j < A->row_size; j++) {
            printf("%f ", get_element(A, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

Matrix* mul_mm(Matrix *A, Matrix *B) {
    if (A->row_size != B->col_size) {
        printf("Wrong matrix sizes\n");
        return NULL;
    }
    Matrix *C = create_zero_matrix(A->col_size, B->row_size);
    if (C == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < A->col_size; i++) {
        for (size_t k = 0; k < A->row_size; k++) {
            for (size_t j = 0; j < B->row_size; j++) {
                double new_value = get_element(C, i, j) + get_element(A, i, k) * get_element(B, k, j);
                set_element(C, i, j, new_value);
            }
        }
    }
    return C;
}

void subtract_mm(Matrix *A, Matrix *B) {
    for (size_t i = 0; i < A->col_size; i++) {
        for (size_t j = 0; j < A->row_size; j++) {
            set_element(A, i, j, get_element(A, i, j) - get_element(B, i, j));
        }
    }
}

void mul_m_number(Matrix *A, double n) {
    for (size_t i = 0; i < A->col_size; i++) {
        for (size_t j = 0; j < A->row_size; j++) {
            set_element(A, i, j, get_element(A, i, j) * n);
        }
    }
}

double modulus(Matrix *A) {
    if (A->row_size != 1) {
        printf("Wrong argument\n");
        return -1;
    }
    double result = 0;
    for (size_t i = 0; i < A->col_size; i++) {
        result += get_element(A, i, 0) * get_element(A, i, 0);
    }

    return sqrt(result);
}

int solve_eq(Matrix *A, Matrix *x, Matrix *b) {
    size_t hit_counter = 0;
    size_t divergence_counter = 0;
    double prev_hit = 0;
    int thao_sign = 1;

    do {
        Matrix *TMP = mul_mm(A, x);
        if (TMP == NULL) {
            free_matrix(A);
            free_matrix(x);
            free_matrix(b);
            return 1;
        }
        subtract_mm(TMP, b);
        double hit = modulus(TMP) / modulus(b);
        mul_m_number(TMP, thao_sign * thao);
        subtract_mm(x, TMP);

        //printf("hit: %f\n", hit);

        if (hit < epsilon) {
            hit_counter++;
        }
        else {
            hit_counter = 0;
        }

        if (hit > prev_hit) {
            divergence_counter++;
        }
        else {
            divergence_counter = 0;
        }

        if (divergence_counter > 5) {
            if (hit == INFINITY || abs(hit) == NAN) {
                printf("Can't solve this\n");
                free_matrix(TMP);
                return 1;
            }
            thao_sign = -thao_sign;
            divergence_counter = 0;
        }
        prev_hit = hit;
        free_matrix(TMP);
    } while (hit_counter < 5);

    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Wrong number of args\n");
        return 1;
    }

    struct timespec start, end;

    size_t N = atol(argv[1]);
    Matrix *x = create_random_matrix(N, 1);
    if (x == NULL) {
        return 1;
    }
    Matrix *b = create_random_matrix(N, 1);
    if (b == NULL) {
        free_matrix(x);
        return 1;
    }
    Matrix *A = create_uninit_matrix(N, N);
    if (A == NULL) {
	    free_matrix(x);
	    free_matrix(b);
	    return 1;
    }
    fill_random_sym_matrix(A);
    
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    if (solve_eq(A, x, b) < 0) {
        free_matrix(A);
        free_matrix(x);
	    free_matrix(b);
	    return 1;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    Matrix* Ax = mul_mm(A, x);
    printf("Ax:\n");
    print_matrix(Ax);
    printf("b:\n");
    print_matrix(b);

    printf("1st programm\n");
    printf("Time taken: %f sec\n\n", end.tv_sec - start.tv_sec + 0.000000001 * (end.tv_nsec - start.tv_nsec));

    free_matrix(A);
    free_matrix(x);
    free_matrix(b);
    //free_matrix(Ax);

    return 0;
}