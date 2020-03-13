//
// Qmn.cpp : Calculates all elements of the Qmn matrix
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <pthread.h>

// Constants of the Dicke model
const double omega = 1
const double omega0 = 1
const double lambda = 1;
const double lambdac = 0.5*sqrt(omega*omega0);

// Diagonalization parameters
const int j = 50;
const int nmax = 300;
const double G2 = 4.0 * lambda / (sqrt(2.0*j)*omega);	// Parameter of the model

const int numT = 600;									// Number of time steps
const double startT = -1;								// First time
const double stepT = 0.01;								// Size of the time step

// Number of eigenvalues
const int numev = 22548;

// Number of used threads
const int threads = 8;

#define QMN "./j50/qks_f_2._j_50_nmax_300_d_30401_dc_22548.dat"
#define EVALUES "./j50/EN_f_2._j_50_nmax_300_d_30401_dc_22548.dat"
#define RESULT "./j50motocs/OTOC_f_2._j_50_nmax_300_k_%d_T_1E6_20000_128_pq.dat"
	 
void Error(const char *error) {
	printf("%s", error);
	exit(0);
}

// Imports Qmn elements and Eigenvalues
void Import(double *qmn, double *eval) {
	time_t start = time(0);
	printf("Importing data...");

	FILE *f = fopen(QMN, "r");
	for (long i = 0; i < numev*numev; i++) {
		if(fscanf(f, "%lf", qmn + i) <= 0)
			Error("Error reading file");
			
	}
	fclose(f);

	f = fopen(EVALUES, "r");
	for (int i = 0; i < numev; i++) {
		if(fscanf(f, "%lf", eval + i) <= 0)
			Error("Error reading file");
	}
	fclose(f);
	printf("%ld\n", time(0) - start);
}

// Saves the result
void Save(char *fname, double *ts, double *result) {
	time_t start = time(0);
	printf("Saving results into %s...", fname);

	FILE *f = fopen(fname, "w");
	for (long i = 0; i < numT; i++) {
		fprintf(f, "%E\t%E\n", ts[i], result[i]);
	}
	fclose(f);

	printf("%ld\n", time(0) - start);
}

struct ThreadData {
	double *result;
	double *qmn;
	double *eval;
	double *ts;
	int i;
	int n;
};

void *Thread(void *data) {
	ThreadData *td = (ThreadData *)data;

	double *eval = td->eval;
	double *result = td->result;
	double *qmn = td->qmn;
	double *ts = td->ts;
	int n = td->n;

	for(int ti = td->i; ti < numT; ti += threads) {
		time_t start = time(0);
		double t = ts[ti];

		double dr = 0;
		double di = 0;
		
		for (int i = 0; i < numev; i++) {
			double a = qmn[n*numev + i];
			double de1 = eval[n] - eval[i];
			for (int j = 0; j < numev; j++) {
				double b = a * qmn[i*numev+j];
				double de2 = eval[i] - eval[j];
			
				dr += b * (de2 * cos(de1 * t) - de1 * cos(de2 * t));
				di += b * (de2 * sin(de1 * t) - de1 * sin(de2 * t));
			}
		}
		
		result[ti] = dr*dr + di*di;
		printf("Time %lf (%d) in thread %d calculated in %lds. OTOC = %lf\n", t, ti, td->i, time(0) - start, result[ti]);
	}

	printf("Thread %d has finished.\n", td->i);
	pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
	setbuf(stdout, 0);				// Just to write a message before a line break

	int n = -1;						// Index of the energy we want to calculate [Watch out, C++ starts indexing from 0, while Mathematica from 1]
	if(argc > 1)
		n = atoi(argv[1]);

	if(n < 0 || n >= numev)
		Error("Invalid level to calculate.\n");

	double *qmn = new double[numev*numev];
	double *eval = new double[numev];
	Import(qmn, eval);

	double *ts = new double[numT];
	for(int i = 0; i < numT; i++)
		ts[i] = pow(10, startT + stepT*i);

	time_t start = time(0);
	printf("Computing MOTOC for energy E (%d) = %lf in %d threads...\n", n, eval[n], threads);

	double *result = new double[numT];

	pthread_t *threadid = new pthread_t[threads];
	ThreadData *td = new ThreadData[threads];

	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for (int i = 0; i < threads; i++) {
		td[i].i = i;
		td[i].result = result;
		td[i].qmn = qmn;
		td[i].eval = eval;
		td[i].ts = ts;
		td[i].n = n;

		pthread_create(threadid + i, NULL, Thread, td + i);
	}

	pthread_attr_destroy(&attr);

	void *status;
	for (int i = 0; i < threads; i++) {
		pthread_join(threadid[i], &status);
	}
	printf("\nCalculation finished in %ld seconds.\n", time(0) - start);

	char fname[1024];
	sprintf(fname, RESULT, n);

	Save(fname, ts, result);
	
	delete(qmn);
	delete(eval);
	delete(ts);
}