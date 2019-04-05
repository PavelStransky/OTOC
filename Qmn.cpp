//
// Qmn.cpp : Calculates all elements of the Qmn matrix
//

#include <stdio.h>
#include <math.h>
#include <ctime>
#include <pthread.h>

// Constants of the Dicke model
double omega = 0.5;
double omega0 = 0.64899375722968400992;
double lambda = 0.66;

#define EVECTORS "./EV_f_2.31723_j_100_nmax_300_d_60501_dc_16710.dat"
#define EVALUES "./EN_f_2.31723_j_100_nmax_300_d_60501_dc_16710.dat"
#define OUTPUT "./Qmn_f_2.31723_j_100_nmax_300_d_60501_dc_16710.dat"

// Diagonalization parameters
int j = 100;
int nmax = 300;
int sz;			// Number of eigenvector elements
double G2;		// Parameter of the model

// Number of eigenvalues eigenvalues
int numev = 16710;

int threads = 32;

// Index
inline int Index(int n, int m) {
	return n*(2 * j + 1) + m + j;
}

void Import(double *evec, double *eval) {
	time_t start = time(0);
	printf("Importing data...");

	FILE *f = fopen(EVECTORS, "r");
	for (long i = 0; i < sz*numev; i++) {
		fscanf(f, "%lf", evec + i);
	}
	fclose(f);

	f = fopen(EVALUES, "r");
	for (int i = 0; i < numev; i++) {
		fscanf(f, "%lf", eval + i);
	}
	fclose(f);
	printf("%ld\n", time(0) - start);
}

void Save(double *result) {
	time_t start = time(0);
	printf("Saving results...");

	FILE *f = fopen(OUTPUT, "w");
	for (long i = 0; i < numev*numev; i++) {
		fprintf(f, "%E", result[i]);
		if ((i + 1) % numev == 0)
			fprintf(f, "\n");
		else
			fprintf(f, "\t");
	}
	fclose(f);

	printf("%ld\n", time(0) - start);
}

struct ThreadData {
	double *result;
	double *evec;
	int i;
};

void *Thread(void *data) {
	ThreadData *td = (ThreadData *)data;

	double *evec = td->evec;
	double *result = td->result;

	for (int ik = td->i; ik < numev; ik += threads) {
		time_t start = time(0);
		int k = ik*sz;
		for (int is = ik; is < numev; is++) {
			int s = is*sz;
			double r = 0.0;
			for (int ni = 0; ni < nmax; ni++) {
				for (int mj = -j; mj <= j; mj++) {
					double d = -G2 * mj * evec[s + Index(ni, mj)];
					if (ni < nmax - 1)
						d += sqrt(ni + 1.0) * evec[s + Index(ni + 1, mj)];
					if (ni > 0)
						d += sqrt((double)ni)*evec[s + Index(ni - 1, mj)];

					r += evec[k + Index(ni, mj)] * d;
				}
			}
			result[ik*numev + is] = r;
			result[is*numev + ik] = r;
		}
		printf("Row %d (%d) calculated in %ld s\n", ik, td->i, time(0) - start);
	}
	printf("Thread %d has finished.\n", td->i);
	pthread_exit(NULL);
}


int main(int argc, char* argv[]) {
	setbuf(stdout, 0);

	sz = Index(nmax, j) + 1;

	double *evec = new double[sz*numev];
	double *eval = new double[numev];
	Import(evec, eval);

	G2 = 4.0 * lambda / (sqrt(2.0*j)*omega);

	int s = 0;
	int k = 0;

	time_t start = time(0);
	printf("Computing in %d threads...\n", threads);

	double *result = new double[numev*numev];

	pthread_t *threadid = new pthread_t[threads];
	ThreadData *td = new ThreadData[threads];

	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for (int i = 0; i < threads; i++) {
		td[i].i = i;
		td[i].result = result;
		td[i].evec = evec;

		pthread_create(threadid + i, NULL, Thread, td + i);
	}

	pthread_attr_destroy(&attr);

	void *status;
	for (int i = 0; i < threads; i++) {
		pthread_join(threadid[i], &status);
	}
	printf("\nCalculation finished in %ld seconds.\n", time(0) - start);

	Save(result);
}
