/* CDLoop.c
*  Perform coordinate descent loop for the two-graph guided multi-task Lasso.
*  Author: Xiaohui Chen (xiaohuic@ece.ubc.ca), Xing Xu (xing@ttic.edu)
*  Last update: Oct-26-2011
*/

#include <mex.h>
#include <math.h>
#include <omp.h>

#define CHUNKSIZE 100
#define EPOWER 4

int sign(double x)
{
    if(x > 0) return 1;
    if(x < 0) return -1;
    return 0;
}

void updateB( double *X, double *Y, double *B, double D_s, double *W1, double *C1, double D1_s, int *E1, 
            double *W2, double *C2, double D2_s, int *E2, double lambda, double gamma1, double gamma2, 
            int J, int K, int n, double *B_new, int num_edges1, int num_edges2)
{
    int j, k, e, i, m, l, f, g, sice;
    double tmp, epsilon;
	double *B_up, *B_down, *R;

	B_up = (double *)malloc(J * K * sizeof(double));
	B_down = B_new;
	epsilon = 1 / pow(n, EPOWER);

	#pragma omp parallel for schedule(dynamic, CHUNKSIZE) private(i, j, k, tmp, R)
			for (k = 0; k < K; k ++)
			{
				R = (double *)malloc(n * sizeof(double));
				for (i = 0; i < n; i++)
				{
					tmp = 0;
					for (j = 0; j < J; j++)
					{
						tmp += X[j*n+i] * B[k*J+j];
					}
					R[i] = tmp;
				}
				for (j = 0; j < J; j ++)
				{
					tmp = 0;
					for (i = 0; i < n; i++)
					{
						tmp += X[j*n+i] * (Y[k*n+i] + X[j*n+i]*B[k*J+j] - R[i]);
					}
					B_up[k*J+j] = tmp;
				}
				free(R);
			}

	#pragma omp parallel for schedule(dynamic, CHUNKSIZE) private(i, j, k, tmp)
			for (j = 0; j < J; j++)
			{
				tmp = 0;
				for (i = 0; i < n; i++)
				{
					tmp += pow(X[j*n+i], 2);
				}
				for (k = 0; k < K; k++)
				{
					B_down[k*J+j] = tmp + lambda * D_s / (fabs(B[k*J+j]) + epsilon);
				}
			}

	#pragma omp parallel for schedule(dynamic, CHUNKSIZE) private(e, sice, j, k, tmp)
			for (e = 0; e < num_edges1; e++)
			{
				sice = sign(C1[e]);
				m = E1[2*e];
				k = E1[2*e+1];
				for (j = 0; j < J; j++)
				{
					tmp = gamma1 * W1[e] * D1_s / (fabs(B[m*J+j] - sice*B[k*J+j]) + epsilon);
					B_down[k*J+j] += tmp;
					B_up[k*J+j] += tmp * B[m*J+j] * sice;
				}

				k = E1[2*e];
				l = E1[2*e+1];
				for (j = 0; j < J; j++)
				{
					tmp = gamma1 * W1[e] * D1_s / (fabs(B[k*J+j] - sice*B[l*J+j]) + epsilon);
					B_down[k*J+j] += tmp;
					B_up[k*J+j] += tmp * B[l*J+j] * sice;
				}
			}

	#pragma omp parallel for schedule(dynamic, CHUNKSIZE) private(e, sice, j, k, tmp)
			for (e = 0; e < num_edges2; e++)
			{
				sice = sign(C2[e]);
				f = E2[2*e];
				j = E2[2*e+1];
				for (k = 0; k < K; k++)
				{
					tmp = gamma2 * W2[e] * D2_s / (fabs(B[k*J+f] - sice*B[k*J+j]) + epsilon);
					B_down[k*J+j] += tmp;
					B_up[k*J+j] += tmp * B[k*J+f] * sice;
				}

				j = E2[2*e];
				g = E2[2*e+1];
				for (k = 0; k < K; k++)
				{
					tmp = gamma2 * W2[e] * D2_s / (fabs(B[k*J+j] - sice*B[k*J+g]) + epsilon);
					B_down[k*J+j] += tmp;
					B_up[k*J+j] += tmp * B[k*J+g] * sice;
				}
			}
	
	for (k = 0; k < K; k++)
	{
		for (j = 0; j < J; j++)
		{
			B_new[k*J+j] = B_up[k*J+j] / B_down[k*J+j];
		}
	}

	free(B_up);
}

double getDSum( double * B, int J, int K, int n )
{
    int j, k;
    double s = 0;
        
    for(k=0;k<K;k++)
    {
        for(j=0;j<J;j++)
        {
            s += fabs( B[k*J+j] );
        }
    }

	return s + J * K * 1 / pow(n, EPOWER);
}

double getD1Sum( double *W1, double *B, 
        double *C1, int *E1, int J, int K, int n, int num_edges1 )
{
    int j, e;
    double s = 0;
    
	for(e=0; e<num_edges1; e++)
    {
		for(j=0; j<J; j++)
		{
			s += fabs( W1[e] *  ( B[E1[2*e]*J+j] - sign(C1[e]) * B[E1[2*e+1]*J+j] ) );
		}
    }

	return s + num_edges1 * J * 1 / pow(n, EPOWER);
}

double getD2Sum( double *W2, double *B, 
        double *C2, int *E2, int J, int K, int n, int num_edges2 )
{
    int k, e;
    double s = 0;
    
	for(e=0; e<num_edges2; e++)
    {
		for(k=0; k<K; k++)
		{
			s += fabs( W2[e] * ( B[k*J+E2[2*e]] - sign(C2[e]) * B[k*J+E2[2*e+1]] ) );
		}
    }

	return s + num_edges2 * K * 1 / pow(n, EPOWER);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *B, *W1, *C1, *W2, *C2, *X, *Y, *B_new, *B_prime, *E1_d, *E2_d;
	int *E1, *E2;
    double lambda, gamma1, gamma2, diff, tol, D1_s, D2_s, D_s;
    int J, K, n, e, flag_it = 0, ind = 0, max_it, num_edges1, num_edges2, tid;
    
    B = mxGetPr(prhs[0]);
    W1 = mxGetPr(prhs[1]);
    C1 = mxGetPr(prhs[2]);
    E1_d = mxGetPr(prhs[3]);
    W2 = mxGetPr(prhs[4]);
    C2 = mxGetPr(prhs[5]);
    E2_d = mxGetPr(prhs[6]);
    X = mxGetPr(prhs[7]);
    Y = mxGetPr(prhs[8]);
    lambda = mxGetScalar(prhs[9]);
    gamma1 = mxGetScalar(prhs[10]);
    gamma2 = mxGetScalar(prhs[11]);
    tol = mxGetScalar(prhs[12]);
    max_it = mxGetScalar(prhs[13]);
    
    diff = tol + 1;
    
    J = mxGetM(prhs[0]);
    K = mxGetN(prhs[0]);
    n = mxGetM(prhs[7]);
	num_edges1 = mxGetN(prhs[3]);
	num_edges2 = mxGetN(prhs[6]);
    
	E1 = (int *)malloc(2 * num_edges1 * sizeof(int));
	E2 = (int *)malloc(2 * num_edges2 * sizeof(int));
	B_prime = (double *)malloc(J * K * sizeof(double));
	plhs[0] = mxCreateDoubleMatrix(J, K, mxREAL);
	B_new = mxGetPr(plhs[0]);
	for (ind = 0; ind < J * K; ind ++)
	{
		*(B_prime + ind) = *(B + ind);
	}
	for (e = 0; e < 2 * num_edges1; e++)
	{
		E1[e] = (int)(E1_d[e] + 0.1);
	}
	for (e = 0; e < 2 * num_edges2; e++)
	{
		E2[e] = (int)(E2_d[e] + 0.1);
	}
    
    while(diff > tol && flag_it < max_it)
    {
#pragma omp parallel
		{
	#pragma omp sections
			{
		#pragma omp section
			    D1_s = getD1Sum(W1, B_prime, C1, E1, J, K, n, num_edges1);
		#pragma omp section
				D2_s = getD2Sum(W2, B_prime, C2, E2, J, K, n, num_edges2);
		#pragma omp section
				D_s = getDSum(B_prime, J, K, n);
			}
		}

		updateB( X, Y, B_prime, D_s, W1, C1, D1_s, E1, W2, C2, D2_s, E2, lambda, gamma1, gamma2, J, K, n, B_new, num_edges1, num_edges2);
        diff = 0;
        for(ind=0; ind < J*K; ind++)
        {
            diff += fabs(B_new[ind] - B_prime[ind]);
            *(B_prime + ind) = *(B_new + ind);
        }
        flag_it++;
    }
    free(B_prime); free(E1); free(E2);
}
