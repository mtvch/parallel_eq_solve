#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mpi/mpi.h"
#include <unistd.h>

typedef struct Matrix Matrix;

struct Matrix {
    double *elements;
    size_t row_size;
    size_t col_size;
    size_t index;
};

const int max_element_value = 20;
const int overload_value = 50;
const double thao = 0.001;
const double epsilon = 0.00001;

double get_element(const Matrix *A, size_t i, size_t j) {
    if (i >= A->col_size || j >= A->row_size) {
        printf("MATRIX SIZE: %lu %lu\n", A->col_size, A->row_size);
        printf("I: %lu, J: %lu\n", i, j);
        printf("Out of range\n");
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

Matrix* create_uninit_matrix(size_t col_size, size_t row_size, size_t index) {
    Matrix *A = malloc(sizeof(Matrix));
    if (A == NULL) {
        printf("Memory error\n");
        return NULL;
    }
    A->elements = malloc(row_size * col_size * sizeof(double));
    if (A->elements == NULL) {
        printf("Memory error: %lu, %lu\n", row_size, col_size);
        return NULL;
    }
    A->col_size = col_size;
    A->row_size = row_size;
    A->index = index;
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

Matrix* create_random_matrix(size_t col_size, size_t row_size, size_t index) {
    Matrix *A = create_uninit_matrix(col_size, row_size, index);
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

Matrix* create_zero_matrix(size_t col_size, size_t row_size, size_t index) {
    Matrix *A = create_uninit_matrix(col_size, row_size, index);
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
    Matrix *C = create_zero_matrix(A->col_size, B->row_size, A->index);
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
            set_element(A, i, j, get_element(A, i, j) - get_element(B, A->index + i, j));
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

int get_portion_size(int n_of_elements, int n_of_blocks, int block_number) {
    int portion_size = n_of_elements / n_of_blocks;

    if (block_number < n_of_elements % n_of_blocks) {
        portion_size++;
    }
    return portion_size;
}

double distributed_modulus(Matrix *A) {
    if (A->row_size != 1) {
        printf("Wrong argument\n");
        return -1;
    }
    double sum = 0;
    double result;
    for (size_t i = 0; i < A->col_size; i++) {
        sum += get_element(A, i, 0) * get_element(A, i, 0);
    }
    
    MPI_Allreduce(&sum, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    return sqrt(result);
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

int build_x(Matrix* x, Matrix *A, int world_size, int rank) {
    Matrix *new_x = create_uninit_matrix(A->col_size, 1, 0);
    if (new_x == NULL) {
        return -1;
    }
    for (size_t i = 0; i < A->col_size; i++) {
        set_element(new_x, i, 0, get_element(x, A->index + i, 0) - get_element(A, i, 0));
    }

    int *recv_counts = malloc(sizeof(int) * world_size);
    if (recv_counts == NULL) {
        printf("Memory error\n");
        free_matrix(new_x);
        return -1;
    }
    for (size_t i = 0; i < world_size; i++) {
        recv_counts[i] = get_portion_size(x->col_size, world_size, i);
    }

    int *displs = malloc(world_size * sizeof(int));
    if (displs == NULL) {
        printf("Memory error\n");
        free_matrix(new_x);
        free(recv_counts);
        return -1;
    } 
    displs[0] = 0;
    for (size_t i = 1; i < world_size; i++) {
        displs[i] = displs[i-1] + recv_counts[i-1];
    }

    if (MPI_Allgatherv(new_x->elements, recv_counts[rank], MPI_DOUBLE, x->elements, recv_counts, displs, MPI_DOUBLE, MPI_COMM_WORLD) != MPI_SUCCESS) {
        printf("ERROR BUILDING X\n");
        free_matrix(new_x);
        free(recv_counts);
        free(displs);
        return -1;
    }

    free(new_x);
    free(recv_counts);
    free(displs);

    return 0;
}

int solve_eq(Matrix* A_part, Matrix* x, Matrix * b, int rank, int world_size) {
    size_t hit_counter = 0;
    size_t divergence_counter = 0;
    double prev_hit = 0;
    int thao_sign = 1;

    do {
        Matrix *TMP = mul_mm(A_part, x);
        if (TMP == NULL) {
            return -1;
        }
        subtract_mm(TMP, b);
        double hit = distributed_modulus(TMP) / modulus(b);
        mul_m_number(TMP, thao_sign * thao);

        if (build_x(x, TMP, world_size, rank) < 0) {
            free_matrix(TMP);
            return -1;
        }        
        // if (rank == 0) {
        //     printf("hit: %f\n", hit);
        // }
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
            thao_sign = -thao_sign;
            divergence_counter = 0;
        }
        if (hit == INFINITY || abs(hit) == NAN) {
            printf("Can't solve this\n");
            free_matrix(TMP);
            return -1;
        }
        prev_hit = hit;
        free_matrix(TMP);
    } while (hit_counter < 5);

    return 0;
}

int distribute_matrix_by_rows(Matrix* A, Matrix* A_part, int rank, int world_size) {
    int *sendcounts = malloc(world_size * sizeof(int));
    if (sendcounts == NULL) {
        printf("Memory error\n");
        return -1;
    }
    for (size_t i = 0; i < world_size; i++) {
        sendcounts[i] = get_portion_size(A->row_size, world_size, i) * A->col_size;
    }

    int *displs = malloc(world_size * sizeof(int));
    displs[0] = 0;
    if (displs == NULL) {
        printf("Memory error\n");
        free(sendcounts);
        return -1;
    }
    for (size_t i = 1; i < world_size; i++) {
        displs[i] = displs[i-1] + sendcounts[i-1];
    }

    A_part->index = displs[rank] / A->row_size;

    if (MPI_Scatterv(A->elements, sendcounts, displs, MPI_DOUBLE, A_part->elements, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
        free(displs);
        free(sendcounts);
        printf("Data distribution error\n");
        return -1;
    }
    free(displs);
    free(sendcounts);
    
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Wrong number of args\n");
        return 1;
    }

    struct timespec start, end;

    size_t N = atol(argv[1]);
    Matrix *A = create_uninit_matrix(N, N, 0);
    if (A == NULL) {
        return 1;
    }
    Matrix *x = create_uninit_matrix(N, 1, 0);
    if (x == NULL) {
        free_matrix(A);
        return 1;
    }
    Matrix *b = create_uninit_matrix(N, 1, 0);
     if (b == NULL) {
        free_matrix(A);
        free_matrix(x);
        return 1;
    }   
        
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        free_matrix(A);
        free_matrix(x);
        free_matrix(b);
        printf("Init error\n");
        return 1;
    }

    int world_size;
    if (MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS) {
        free_matrix(A);
        free_matrix(x);
        free_matrix(b);
        printf("Comm_size error\n");
        return 1;
    }

    int rank;
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS) {
        free_matrix(A);
        free_matrix(x);
        free_matrix(b);
        printf("Comm_rank error\n");
        return 1;
    }

    Matrix *A_part = create_uninit_matrix(get_portion_size(N, world_size, rank), N, 0);
    if (A_part == NULL) {
        free_matrix(x);
        free_matrix(b);
        free_matrix(A);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        fill_random_sym_matrix(A);
        fill_random_matrix(x);
        fill_random_matrix(b);
    }

    if (MPI_Bcast(x->elements, N, MPI_DOUBLE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
        printf("Bcast error\n");
        free_matrix(A_part);
        free_matrix(x);
        free_matrix(b);
        free_matrix(A);
        MPI_Finalize();
        return 1;
    }
    if (MPI_Bcast(b->elements, N, MPI_DOUBLE, 0, MPI_COMM_WORLD) != MPI_SUCCESS) {
        printf("Bcast error\n");
        free_matrix(A_part);
        free_matrix(x);
        free_matrix(b);
        free_matrix(A);
        MPI_Finalize();
        return 1;
    }

    if (distribute_matrix_by_rows(A, A_part, rank, world_size) < 0) {
        free_matrix(A_part);
        free_matrix(x);
        free_matrix(b);
        free_matrix(A);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    }

    if (solve_eq(A_part, x, b, rank, world_size) < 0) {
        free_matrix(A);
        free_matrix(A_part);
        free_matrix(x);
        free_matrix(b);
        MPI_Finalize();

        return 1;
    }
    
    if (rank == 0) {
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    }

    if (rank == 0) {
        Matrix* Ax = mul_mm(A, x);
        printf("Ax:\n");
        print_matrix(Ax);
        printf("b:\n");
        print_matrix(b);
        free_matrix(Ax);
    }

    if (rank == 0) {
        printf("2nd programm %d processes\n", world_size);
        printf("Time taken: %f sec\n\n", end.tv_sec - start.tv_sec + 0.000000001 * (end.tv_nsec - start.tv_nsec));
    }
    
    free_matrix(A);
    free_matrix(x);
    free_matrix(b);
    free_matrix(A_part);

    MPI_Finalize();

    return 0;
}