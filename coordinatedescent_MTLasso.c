/* coordinatedescent_MTLasso.c
*  Perform coordinate descent algorithm for standard multi-task Lasso.
*  Author: Xiaohui Chen (xiaohuic@ece.ubc.ca)
*  Last update: Sep-16-2011
*/

#include "mex.h"
#include "math.h"

int sign(double num)
{
    int s = 0;
    if(num > 0) { s = 1; }
    if(num < 0) { s = -1; }
    return s;
}

void updateB( double *X, double *Y, double *B, double *D, double lambda,
            int J, int K, int n, double *B_new )
{
    int i, j, k, j2;
    double residual, b_up, b_down;
    for(j=0; j<J; j++)
    {
        for(k=0; k<K; k++)
        {
            b_up = 0;   b_down = 0;
            for(i=0; i<n; i++)
            {
				residual = 0;
                for(j2=0; j2<J; j2++)
                {
					if (j2 != j)
					{
						residual += X[j2*n+i] * B[k*J+j2];
					}
                }
                residual = Y[k*n+i] - residual;
                b_up += X[j*n+i] * residual;
            }
            
            for(i=0; i<n; i++)
            {
                b_down += pow(X[j*n+i], 2);
            }
			b_down += lambda / D[j];
            
            *(B_new + k*J + j) = b_up / b_down;
        }
    }
}

void updateD( double * D, double * B, int J, int K, int n )
{
    int j, k, ind;
    double s = 0, *D_tmp = NULL;
    
    D_tmp = (double *)malloc(J * sizeof(double));
    
    for(j=0;j<J;j++)
    {
        D_tmp[j] = 0;
        for(k=0;k<K;k++)
        {
            ind = k*J+j;
            D_tmp[j] += pow( B[ind], 2 );
        }
		D_tmp[j] = sqrt(D_tmp[j]) + 1 / pow(n, 2);
		s += D_tmp[j];
    }
    for(j=0;j<J;j++)
    {
        D[j] = D_tmp[j] / s;
    }
    free(D_tmp);
}


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    double *B, *X, *Y, *D;
    double *B_new, *D_new;
    double lambda;
    int J, K, n;
    
    B = mxGetPr(prhs[0]);
    X = mxGetPr(prhs[1]);
    Y = mxGetPr(prhs[2]);
    D = mxGetPr(prhs[3]);
    lambda = mxGetScalar(prhs[4]);
    
    J = mxGetM(prhs[0]);
    K = mxGetN(prhs[0]);
    n = mxGetM(prhs[1]);
    
    plhs[0] = mxCreateDoubleMatrix(J, K, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(J, 1, mxREAL);
    
    B_new = mxGetPr(plhs[0]);
    D_new = mxGetPr(plhs[1]);
    
    updateB( X, Y, B, D, lambda, J, K, n, B_new );
    updateD( D_new, B_new, J, K, n );
}
