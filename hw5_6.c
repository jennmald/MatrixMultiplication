/* This is the template file for HW5_6. You need to complete the code by 
 * implementing the parts marked by FIXME. 
 *
 * Submit your completed code to the TA by email. Also submit the plots 
 * and your conclusions of the comparative study.
 *
 * To compile, use the UNIX command 
 *      make
 * in the source directory, and it would then invoke the compilation command 
 * with the included makefile. You may sometimes need to use the command 
 *      make clean
 * before recompiling.
 *
 * This will generate an executable hw5_6. The -g option is optional and 
 * is needed only if you need to debug the program in a debugger such as ddd.
 * The -Wall option would enable compiler's warning messages.
 *
 * Run the program with command
 *      ./hw5_6
 * which would generate a M-file, which you can use to generate the plots by
 * issueing the UNIX command 
 *      make plot
 * in the source directory. It will then generate two PDF files, which you
 * should submit to the TA along with your source code and your conclusions.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/* All type definitions and function prototypes are in this header file. */
#include "hw5_6.h"

#define MAXDEGREE 15
#define MAXPOINTS 30
#define MAXMATRIXSIZE  450
#define MAXRSIZE  225

int main(int argc, char **argv) 
{   
    /* Buffer vectors */
    double ts[MAXPOINTS];
    double dtemp[MAXDEGREE];

    /* Arrays for storing error and timing results */
    double err_svd[MAXDEGREE-2], err_eig[MAXDEGREE-2];
    double times_svd[MAXDEGREE-2], times_eig[MAXDEGREE-2];

    FILE *fid;
    int m, n, i, j;
    int niter = 1000;

    /* Set n to degree of polynomial */
    for (n=3; n<=MAXDEGREE; ++n) {
        Matrix A; 
        Vector b,x;

        /* Set m to the number of points */
        m = n + n;

        A = allocate_matrix( m, n);
        b = allocate_vector( m);
        x = allocate_vector( n);

        /* Compute Vandermonde matrix A and right-hand side b */
        for (i=0; i<m; i++) {
            ts[i] = (((double)i)) / (((double)(m - 1)));

            A.vals[i][0] = 1.0;
            b.vals[i] = 1.0;
        }

        for (j=1; j<n; j++) {
            for (i=0; i<m; i++) {
                A.vals[i][j] = A.vals[i][j-1] * ts[i];
                b.vals[i] = 1.0 +  b.vals[i] * ts[i];
            }
        }

        tictoc(1);
        /* Invoke least-squares solver using SVD (dgesvd). */
        /* Run for many iterations to obtain more accurate timing results */
        for (j=0; j<niter; ++j) {
            lsqr_dgesvd(x, A, b);
        }
        times_svd[n - 3] =  tictoc(2) / niter;

        for (j=0; j<n; j++) {
            dtemp[j] =  x.vals[j] - 1.0;
        }
        err_svd[n - 3] = norm2_col(dtemp, n);

        tictoc(1);
        /* Invoke least-squares solver using symmetric eigenvalue decomposition 
           of A'*A (dsyev). */
        /* Run for many iterations to obtain more accurate timing results */
        for (j=0; j<niter; ++j) {
            lsqr_dsyev(x, A, b);
        }
        times_eig[n - 3] =  tictoc(2) / niter;

        for (j=0; j<n; j++) {
            dtemp[j] =  x.vals[j] - 1.0;
        }
        err_eig[n - 3] = norm2_col(dtemp, n);

        deallocate_matrix( A);
        deallocate_vector( b);
        deallocate_vector( x);
    }

    /* Write out results into a Matlab file */
    fid = fopen("results.m", "w");
    write_vec(fid, "err_svd", err_svd, MAXDEGREE-2);
    write_vec(fid, "err_eig  ", err_eig, MAXDEGREE-2);
    write_vec(fid, "times_svd", times_svd, MAXDEGREE-2);
    write_vec(fid, "times_eig", times_eig, MAXDEGREE-2);
    fclose(fid);

    /* plotresults; */
    return 0;
}

/*************************************************************
 *
 * FUNCTION: lsqr_dgesvd
 *
 * Least squares via SVD of A.
 *************************************************************/

void  lsqr_dgesvd(
   Vector x, 
   const Matrix A, 
   const Vector b)
{
    int m = A.m;
    int n = A.n;

    Matrix Acopy = allocate_matrix(m,n);
    Matrix U = allocate_matrix(m,n);
    Matrix VT = allocate_matrix(n,n);

    double work[5*MAXPOINTS];
    double S[MAXPOINTS];

    int info;
    int lwork = 5*n;
    int i, j, k;

    /* Make a copy of A to avoid overwriting by dgesvd */
    for (i=0; i<m; ++i) {
        for (j=0; j<n; ++j) {
            Acopy.vals[i][j] = A.vals[i][j];
        }
    }
    
    /* Perform SVD of A.'
       Since A is stored as row-major, it is A' in Fortran convenction. 
       Therefore, we call dgesvd as A'=V S U' to compute SVD, and  
       V is then V' in C convention and U' is U in C convention. */
    dgesvd_("A", "S", &n, &m, &Acopy.vals[0][0], &n, S, &VT.vals[0][0],
            &n, &U.vals[0][0], &n, work, &lwork, &info);

    if (info != 0) {
        printf("ERROR: dgesvd has failed.");
        return;
    }

    /* Compute U*b */
    Vector Ub = allocate_vector(m);
    Ub = U*b;

    /* Solve the diagonal system */
    Vector w = allocate_vector(m);
    w = backsolve(S,U,b);

    /* Set x = Vx */
    x = V*x;
    deallocate_matrix(U);
    deallocate_matrix(VT);
    deallocate_matrix(Acopy);

    return;
}


/*************************************************************
 *
 * FUNCTION: lsqr_dsyev
 *
 * Least squares via symmetric eigenvalue decomposition of A'*A.
 *************************************************************/

void  lsqr_dsyev(
   Vector x, 
   const Matrix A, 
   const Vector b)
{
    int m = A.m;
    int n = A.n;
    Matrix AT = allocate_matrix(n,m);
    Matrix Q = allocate_matrix(m,m);
    Matrix L = allocate_matrix(m,n);
    /* find A transpose */
    for (i=0; i<n; ++i) {
        for (j=0; j<m; ++j) {
            AT.vals[j][i] = A.vals[i][j];
        }
    }

    /* Find A transpose A */
    Matrix ATA = allocate_matrix(m,m);
    ATA = AT * A;

    /* find the eigenvals of AtA */ 
    dsyev_("A", "ATA" , &n, &m, &n, &A.vals[0][0], work, &lwork, &info);
    Vector x = allocate_vector(m);
    /* Find x */
    x = Q*L*Q*AT*b;
    return;
}

/*************************************************************
 *
 * FUNCTION: norm2_col
 *
 * Compute 2-norm of a given vector.
 * Note: It is not a perfect implementation as it does not guard against 
 *       overflow or underflow.
 *************************************************************/

double norm2_col(
    double *vec, 
    int vec_dim)
{
    double v_out;
    int i;

    v_out = 0.0;
    for (i=0; i<vec_dim; i++) {
        v_out = v_out +  vec[i] *  vec[i];
    }
    return sqrt(v_out);
}

/*-----------------------------------------------------------------------------
 * FUNCTION: allocate_matrix - Allocate memory for a given matrix.
 * --------------------------------------------------------------------------*/
/* Disclaimer: The approach used here is not the most efficient way for
 * implementing matrices, but it is adopted here for convenience.
 */
Matrix allocate_matrix(int m, int n) {
    Matrix A;
    double *ptmp;
    int i;

    A.m = m; A.n = n;
    ptmp = (double *)malloc( sizeof(double) * m * n);
    A.vals = (double **)malloc( sizeof(double **) * m);

    for (i=0; i<m; ++i) {
        A.vals[i] = ptmp + i*n;
    }

    return A;
}

/*-----------------------------------------------------------------------------
 * FUNCTION: deallocate_matrix - De-allocate memory for a matrix.
 * --------------------------------------------------------------------------*/
void deallocate_matrix( Matrix A) {
    free( A.vals[0]);
    free( A.vals); A.vals=NULL;
}


/*-----------------------------------------------------------------------------
 * FUNCTION: allocate_vector - Allocate memory for an m vector.
 * --------------------------------------------------------------------------*/
Vector allocate_vector( int m) {
    Vector x;

    x.m = m;
    x.vals = (double *)malloc( sizeof(double) * m);
    return x;
}

/*-----------------------------------------------------------------------------
 * FUNCTION: deallocate_vector - De-allocate the memory for an m vector. 
 * --------------------------------------------------------------------------*/
void deallocate_vector( Vector x) {
    free( x.vals); x.vals=NULL;
}

#include <time.h>
#include <sys/time.h>

/*-----------------------------------------------------------------------------
 * FUNCTION: tictoc - Initialize (1) or get (2) the elapsed time since in seconds
 * --------------------------------------------------------------------------*/
double tictoc (int n)
{
    double    y = 0.0;
    static    struct timeval start_time, end_time; 

    if (n == 1) { 
        gettimeofday(&start_time, NULL);
    }
    else if (n == 2) { 
        gettimeofday(&end_time, NULL);

        y = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec-start_time.tv_usec)*1.e-6;
    }
    return (y); 
}

/*************************************************************
 *
 * FUNCTION: write_vec
 *
 * Write out a vector into file
 *************************************************************/

void  write_vec(FILE *fid, char *name, const double *vec, int n) {
    int i;

    fprintf(fid, "%s=[", name);
    for (i=0; i<n; ++i) {
        fprintf(fid, "%g; ", vec[i]);
    }
    fprintf(fid, "];\n");
}
