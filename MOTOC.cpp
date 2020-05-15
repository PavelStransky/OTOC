//
// Qmn.cpp : Calculates all elements of the Qmn matrix
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <pthread.h>

// Constants of the Dicke model
const double omega = 1;
const double omega0 = 1;
const double lambda = 1;
const double lambdac = 0.5*sqrt(omega*omega0);

const int numTimes = 4000;								// Number of time steps

// Diagonalization parameters
const int j = 50;
const int nmax = 300;

// Number of eigenvalues
const int numEV = 22548;

// Number of used threads
const int threads = 24;

#define THRESHOLD 1E-6

#define QMN "./j50/qks_f_2._j_50_nmax_300_d_30401_dc_22548.dat"
#define EVALUES "./j50/EN_f_2._j_50_nmax_300_d_30401_dc_22548.dat"
#define RESULT "./j50motocs/OTOC_f_2._j_50_nmax_300_k_%d_pq.dat"
	 
void Error(const char *error) {
	printf("%s", error);
	exit(0);
}

// Imports Qmn elements and Eigenvalues
void Import(double *qmn, double *eigenValues) {
	time_t start = time(0);
	printf("Importing data...");

	FILE *f = fopen(QMN, "r");
	for (long i = 0; i < numEV * numEV; i++) {
		if(fscanf(f, "%lf", qmn + i) <= 0)
			Error("Error reading file");
			
	}
	fclose(f);

	f = fopen(EVALUES, "r");
	for (int i = 0; i < numEV; i++) {
		if(fscanf(f, "%lf", eigenValues + i) <= 0)
			Error("Error reading file");
	}
	fclose(f);
	printf("%ld s.\n", time(0) - start);
}

// Saves the result
void Save(char *fname, double *times, double *result) {
	time_t start = time(0);

	FILE *f = fopen(fname, "w");
	for (long i = 0; i < numTimes; i++) {
		fprintf(f, "%E\t%E\n", times[i], result[i]);
	}
	fclose(f);
}

struct ThreadData {
	double *qmn;
	double *eigenValues;
	double *times;
	int i;                  // Thread number
};

void *Thread(void *data) {
	ThreadData *td = (ThreadData *)data;

	double *eigenValues = td->eigenValues;
	double *qmn = td->qmn;
	double *times = td->times;

	double *result = new double[numTimes];
    double *dr = new double[numTimes];
    double *di = new double[numTimes];

    for(int n = numEV - td->i - 1; n >= 0; n -= threads) {
		time_t start = time(0);

        for(int ti = 0; ti < numTimes; ti++) {
            result[ti] = 0.0;
            dr[ti] = 0.0;
            di[ti] = 0.0;
        }

        for (int i = 0; i < numEV; i++) {
            for (int j = 0; j < numEV; j++) {
				double a = qmn[n*numEV + j] * qmn[j*numEV + i];
                if(a > -THRESHOLD && a < THRESHOLD)
                    continue;

	    		double de1 = eigenValues[n] - eigenValues[j];
				double de2 = eigenValues[j] - eigenValues[i];
                
                for(int ti = 0; ti < numTimes; ti++) {
                    double t = times[ti];
              
                    dr[ti] += a * (de2 * cos(de1 * t) - de1 * cos(de2 * t));
                    di[ti] += a * (de2 * sin(de1 * t) - de1 * sin(de2 * t));
                }
            }

            for(int ti = 0; ti < numTimes; ti++) {
                result[ti] += 0.25 * (dr[ti] * dr[ti] + di[ti] * di[ti]);
                dr[ti] = 0.0;
                di[ti] = 0.0;
            }
        }
            
        double sum = 0.0;       // For mean
        double sum2 = 0.0;      // For variance

        for(int ti = numTimes / 2; ti < numTimes; ti++) {
            double d = result[ti];
            sum += d;
            sum2 += d*d;
        }

        double mean = 2 * sum / numTimes;
        double variance = sqrt(2 * sum2 / numTimes - mean * mean);
        double r = variance / mean;

        printf("Thread %d calculated E(%d) = %0.3lf in %ld seconds. OTOC = %0.2lf +- %0.2lf (R = %0.3lf)\n", td->i, n, eigenValues[n], time(0) - start, mean, variance, r);

    	char fname[1024];
	    sprintf(fname, RESULT, n);
    	Save(fname, times, result);
    }

    delete dr, di;
    delete result;

	printf("Thread %d has finished.\n", td->i);
	pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
	setbuf(stdout, 0);				// Just to write a message before a line break

	double *qmn = new double[numEV*numEV];
	double *eigenValues = new double[numEV];
	Import(qmn, eigenValues);

	double *times = new double[numTimes];
    const double stepTime1 = 0.1; 							// Size of the time step
    const double startTime2 = 1E6;	        				// First time
    const double stepTime2 = 1000;

	for(int i = 0; i < numTimes / 2; i++) {
		times[i] = stepTime1 * i;
        times[i + numTimes / 2] = startTime2 + stepTime2 * i;
    }

    time_t start = time(0);
    printf("Computing MOTOC in %d threads, %d time steps...\n", threads, numTimes);

    pthread_t *threadid = new pthread_t[threads];
    ThreadData *td = new ThreadData[threads];

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for (int i = 0; i < threads; i++) {
        td[i].i = i;
        td[i].qmn = qmn;
        td[i].eigenValues = eigenValues;
        td[i].times = times;

        pthread_create(threadid + i, NULL, Thread, td + i);
    }

    pthread_attr_destroy(&attr);

    void *status;
    for (int i = 0; i < threads; i++) {
        pthread_join(threadid[i], &status);
    }
    printf("\nCalculation finished in %ld seconds.\n", time(0) - start);
	
	delete(qmn);
	delete(eigenValues);
	delete(times);

    delete(td);
    delete(threadid);
}
