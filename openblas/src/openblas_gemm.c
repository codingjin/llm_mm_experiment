#include "stdio.h"
#include "stdlib.h"
#include "sys/time.h"
#include "time.h"
#include "cblas.h"

#define ROUND 9
#define ROUND_MEDIAN 5
#define THREAD_NUM_ITER_BOUND 4

int thread_num_settings[] = {0, 8, 16, 20};

int main(int argc, char* argv[])
{
	
	if (argc != 4) {
		printf("Input error, usage: ./openblas_gemm M N K\n");
		return 1;
	}
	
	int thread_num = 0;
	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int k = atoi(argv[3]);
	int sizeofa = m * k;
	int sizeofb = k * n;
	int sizeofc = m * n;

	struct timeval start, finish;
	uint32_t time_array[ROUND];
	uint32_t median_time = 0;

	float* A = (float*)malloc(sizeof(float) * sizeofa);
	float* B = (float*)malloc(sizeof(float) * sizeofb);
	float* C = (float*)malloc(sizeof(float) * sizeofc);

	srand(149);
	for (int i = 0; i < sizeofa; i++)	A[i] = ((float)rand()) / RAND_MAX;
	for (int i = 0; i < sizeofb; i++)	B[i] = ((float)rand()) / RAND_MAX;
	for (int i = 0; i < sizeofc; i++)	C[i] = ((float)rand()) / RAND_MAX;
	
    for (int iter = 0; iter < THREAD_NUM_ITER_BOUND; ++iter) {
        thread_num = thread_num_settings[iter];
        if (thread_num) openblas_set_num_threads(thread_num);

        for (int round = 0; round < ROUND; round++) {
            gettimeofday(&start, NULL);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B, n, 1, C, n);
            gettimeofday(&finish, NULL);
            time_array[round] = ((uint32_t)(finish.tv_sec - start.tv_sec))*1000000 + (uint32_t)(finish.tv_usec - start.tv_usec);
        }

        // Get the median time
        int min_index;
        uint32_t temp, min;
        for (int i = 0; i < ROUND_MEDIAN; ++i) {
            min = time_array[i];
            min_index = i;
            for (int j = i+1; j < ROUND; ++j) {
                if (time_array[j] < min) {
                    min = time_array[j];
                    min_index = j;
                }
            }
            if (min_index != i) {
                temp = time_array[i];
                time_array[i] = min;
                time_array[min_index] = temp;
            }
        }
        median_time = time_array[ROUND_MEDIAN-1];

        printf("ThreadNum=%d M=%d N=%d K=%d, Performance: %.0f GFLOPs/s\n\n", thread_num, m, n, k, 2.0e-3*m*n*k / median_time);
    }

	free(A);
	free(B);
	free(C);
	return 0;
}