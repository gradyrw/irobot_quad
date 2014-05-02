#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include <lwpr.h>
#include <math.h>
#include <lwpr_xml.h>
#define CONTROL_DIM 2
#define STATE_DIM 5
#define DERIV_STATE_DIM 3
//N is the number of states in an LWPR model, in this case 4
#define N 5
#define K 1000
#define M 16
#define T 60
#define HZ 20
#define MAX_VAR 10.0

#define BLOCKSIZE 1024

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//Define a data structure which contains the elements
//of an LWPR receptive field needed to make a prediction.
typedef struct {
  float c[N];
  float D[N*N];
  int trustworthy;
  float beta0;
  float mean_x[N];
  int nReg;
  float n_data[N];
  float U[N*N];
  float P[N*N];
  float beta[N];
  float SSs2[N];
  float sum_e_cv2[N];
  float sum_W[N];
  float SSp;
} RF_Predict;

//Transfers data from a full receptive field to a (smaller) rfPredict struct
void rfTransfer(LWPR_ReceptiveField *rf_orig, RF_Predict *rf_pred) {
  int i;
  for (i = 0; i < N*N; i++) {
    rf_pred->D[i] = rf_orig->D[i];
    rf_pred->U[i] = rf_orig->U[i];
    rf_pred->P[i] = rf_orig->P[i];
  }
  for (i = 0; i < N; i++) {
    rf_pred->c[i] = rf_orig->c[i];
    rf_pred->mean_x[i] = rf_orig->mean_x[i];
    rf_pred->n_data[i] = rf_orig->n_data[i];
    rf_pred->beta[i] = rf_orig->beta[i];
    rf_pred->SSs2[i] = rf_orig->SSs2[i];
    rf_pred->sum_e_cv2[i] = rf_orig->sum_e_cv2[i];
    rf_pred->sum_W[i] = rf_orig->sum_w[i];
  }
  rf_pred->trustworthy = rf_orig->trustworthy;
  rf_pred->beta0 = rf_orig->beta0;
  rf_pred->nReg = rf_orig->nReg;
  rf_pred->SSp = rf_orig->SSp;
}

//==============================================================================
//----------------------------CUDA FUNCTIONS------------------------------------
//==============================================================================

__constant__ float U_d[T*CONTROL_DIM];
__constant__ float dm_d[T*M*DERIV_STATE_DIM];
__constant__ float norm_in1_d[N];
__constant__ float norm_in2_d[N];
__constant__ float norm_in3_d[N];

__device__ void print_vec(float* A, float* B, int n) {
  printf("\n\n++++++++++++++++++++++++++++++++++++++++++");
  printf("\n Printing A \n");
  for (int i = 0; i < n; i++) {
    printf("  %f  ", A[i]);
  }
  printf("- \n \n -");
  printf("\n Printing B \n");
  for (int i = 0; i < n; i++) {
    printf("  %f  ", B[i]);
  }
  printf("=================================\n");
}

__device__ void rf_to_shared_mem(RF_Predict *rf_s, RF_Predict *rf_g, int idx) {
  //Smaller indices load arrays
  if (idx < N*N) {
    rf_s->D[idx] = rf_g->D[idx];
  }
  else if (idx >= N*N && idx < 2*N*N) {
    rf_s->U[idx-N*N] = rf_g->U[idx-N*N];
  }
  else if (idx >= 2*N*N && idx < 3*N*N) {
    rf_s->P[idx-2*N*N] = rf_g->P[idx-2*N*N];
  }
  //Intermediate indices load vectors
  else if (idx >= 3*N*N && idx < 3*N*N + N) {
    rf_s->c[idx-3*N*N] = rf_g->c[idx-3*N*N];
  }
  else if (idx >= 3*N*N + N && idx < 3*N*N + 2*N) {
    rf_s->mean_x[idx-(3*N*N + N)] = rf_g->mean_x[idx-(3*N*N + N)];
  }
  else if (idx >= 3*N*N + 2*N && idx < 3*N*N + 3*N) {
    rf_s->n_data[idx-(3*N*N + 2*N)] = rf_g->n_data[idx-(3*N*N + 2*N)];
  }
  else if (idx >= 3*N*N + 3*N && idx < 3*N*N + 4*N) {
    rf_s->beta[idx-(3*N*N + 3*N)] = rf_g->beta[idx-(3*N*N + 3*N)];
  }
  else if (idx >= 3*N*N + 4*N && idx < 3*N*N + 5*N) {
    rf_s->SSs2[idx-(3*N*N + 4*N)] = rf_g->SSs2[idx-(3*N*N + 4*N)];
  }
  else if (idx >= 3*N*N + 5*N && idx < 3*N*N + 6*N) {
    rf_s->sum_e_cv2[idx-(3*N*N + 5*N)] = rf_g->sum_e_cv2[idx-(3*N*N + 5*N)];
  }
  else if (idx >= 3*N*N + 6*N && idx < 3*N*N + 7*N) {
    rf_s->sum_W[idx-(3*N*N + 6*N)] = rf_g->sum_W[idx-(3*N*N + 6*N)];
  }
  //Big indices load scalars
  else if (idx == 3*N*N + 7*N) {
    rf_s->trustworthy = rf_g->trustworthy;
  }
  else if (idx == 3*N*N + 7*N + 1) {
    rf_s->beta0 = rf_g->beta0;
  }
  else if (idx == 3*N*N + 7*N + 2) {
    rf_s->nReg = rf_g->nReg;
  }
  else if (idx == 3*N*N + 7*N + 3) {
    rf_s->SSp = rf_g->SSp;
  }
}

__device__ void compute_proj(int nR, float* s, float* xc, float* U, float* P) {
  int i,j;
  float dot;
  float xu[N];
  for (i = 0; i < N; i++) {
    xu[i] = xc[i];
  }
  for (i = 0; i < nR - 1; i++) {
    dot = 0;
    for (j = 0; j < N; j++) {
      dot += U[i*N + j]*xu[j];
    }
    s[i] = dot;
    for (j = 0; j < N; j++) {
      xu[j] -= s[i]*P[i*N + j];
    }
  }  
  dot = 0;
  for (i = 0; i < N; i++) {
    dot += U[(nR - 1)*N + i]*xu[i];
  }
  s[nR - 1] = dot;
}

__device__ void rf_predict(RF_Predict *rf, float* pred_helper, float* x, int index) {
  int i,j;
  float xc[N];
  for (i = 0; i < N; i++) {
    xc[i] = x[i] - rf->c[i];
  }
  float dist = 0;
  for (i = 0; i < N; i++) {
    float dot = 0;
    for (j = 0; j < N; j++) {
      dot += rf->D[j*N + i]*xc[j];
    }
    dist += xc[i]*dot;
  }
  float w = __expf(-.5*dist);
  float yp_n;
  float sigma2;
  if (w > .001 && rf->trustworthy) {
    yp_n = rf->beta0;
    sigma2 = 0.0;
    for (i = 0; i < N; i++) {
      xc[i] = x[i] - rf->mean_x[i];
    }
    int nR = rf->nReg;
    if (rf->n_data[nR-1] <= 2*N) {
      nR--;
    }
    float s[N];
    compute_proj(nR, s, xc, rf->U, rf->P);
    for (i = 0; i < nR; i++) {
      yp_n += s[i]*rf->beta[i];
      sigma2 += s[i]*s[i] / rf->SSs2[i];
    }
    sigma2 = rf->sum_e_cv2[nR-1]/(rf->sum_W[nR-1] - rf->SSp)*(1+w*sigma2);
    pred_helper[0] = yp_n*w;
    pred_helper[1] = w;
    pred_helper[2] = w*yp_n*yp_n;
    pred_helper[3] = w*sigma2;
  }
  else {
      pred_helper[0] = 0;
      pred_helper[1] = 0;
      pred_helper[2] = 0;
      pred_helper[3] = 0;
  }
}

__device__ void compute_predict_conf(RF_Predict* rfs, float* x, int numRFS, float* vals) {
  int i;
  float pred_helper[] = {0,0,0,0};
  float sum_wy = 0;
  float sum_w = 0;
  float sum_wyy = 0;
  float sum_conf = 0;
  __shared__ RF_Predict rf_s0;
  __shared__ RF_Predict rf_s1;
  __shared__ RF_Predict rf_s2;
  __shared__ RF_Predict rf_s3;
  __shared__ RF_Predict rf_s4;
  __shared__ RF_Predict rf_s5;
  __shared__ RF_Predict rf_s6;
  __shared__ RF_Predict rf_s7;
  int tot_el = 3*N*N + 7*N + 4;
  int idx = threadIdx.x*M + threadIdx.y;
  for (i = 0; i < numRFS; i+= 8) {    
    __syncthreads();
    if (idx < tot_el && i < numRFS) {
      rf_to_shared_mem(&rf_s0, &rfs[i], idx);
    }
    else if (idx >= tot_el && idx < 2*tot_el && i + 1 < numRFS) {
      rf_to_shared_mem(&rf_s1, &rfs[i+1], idx - tot_el);
    }
    else if (idx >= 2*tot_el && idx < 3*tot_el && i + 2 < numRFS) {
      rf_to_shared_mem(&rf_s2, &rfs[i+2], idx - 2*tot_el);
    }
    else if (idx >= 3*tot_el && idx < 4*tot_el && i + 3 < numRFS) {
      rf_to_shared_mem(&rf_s3, &rfs[i+3], idx - 3*tot_el);
    }
    else if (idx >= 4*tot_el && idx < 5*tot_el && i + 4 < numRFS) {
      rf_to_shared_mem(&rf_s4, &rfs[i+4], idx - 4*tot_el);
    }
    else if (idx >= 5*tot_el && idx < 6*tot_el && i + 5 < numRFS) {
      rf_to_shared_mem(&rf_s5, &rfs[i+5], idx - 5*tot_el);
    }
    else if (idx >= 6*tot_el && idx < 7*tot_el && i + 6 < numRFS) {
      rf_to_shared_mem(&rf_s6, &rfs[i+6], idx - 6*tot_el);
    }
    else if (idx >= 7*tot_el && idx < 8*tot_el && i + 7 < numRFS) {
      rf_to_shared_mem(&rf_s7, &rfs[i+7], idx - 7*tot_el);
    }
    __syncthreads();
    rf_predict(&rf_s0, pred_helper, x, i);
    sum_wy += pred_helper[0];
    sum_w += pred_helper[1];
    sum_wyy += pred_helper[2];
    sum_conf += pred_helper[3];

    if (i + 1 < numRFS) {
      rf_predict(&rf_s1, pred_helper, x, i+1);
      sum_wy += pred_helper[0];
      sum_w += pred_helper[1];
      sum_wyy += pred_helper[2];
      sum_conf += pred_helper[3];
    }
    if (i + 2 < numRFS) {
      rf_predict(&rf_s2, pred_helper, x, i+2);
      sum_wy += pred_helper[0];
      sum_w += pred_helper[1];
      sum_wyy += pred_helper[2];
      sum_conf += pred_helper[3];
    }
    if (i + 3 < numRFS) {
      rf_predict(&rf_s3, pred_helper, x, i+3);
      sum_wy += pred_helper[0];
      sum_w += pred_helper[1];
      sum_wyy += pred_helper[2];
      sum_conf += pred_helper[3];
    }
    if (i + 4 < numRFS) {
      rf_predict(&rf_s4, pred_helper, x, i+4);
      sum_wy += pred_helper[0];
      sum_w += pred_helper[1];
      sum_wyy += pred_helper[2];
      sum_conf += pred_helper[3];
    }
    if (i + 5 < numRFS) {
      rf_predict(&rf_s5, pred_helper, x, i+5);
      sum_wy += pred_helper[0];
      sum_w += pred_helper[1];
      sum_wyy += pred_helper[2];
      sum_conf += pred_helper[3];
    }
    if (i + 6 < numRFS) {
      rf_predict(&rf_s6, pred_helper, x, i+6);
      sum_wy += pred_helper[0];
      sum_w += pred_helper[1];
      sum_wyy += pred_helper[2];
      sum_conf += pred_helper[3];
    }
    if (i + 7 < numRFS) {
      rf_predict(&rf_s7, pred_helper, x, i+7);
      sum_wy += pred_helper[0];
      sum_w += pred_helper[1];
      sum_wyy += pred_helper[2];
      sum_conf += pred_helper[3];
    }
  }
  if (sum_w > 0) {
    vals[0] = sum_wy/sum_w;
    vals[1] = fmin((float)sqrt(fabs(sum_conf + sum_wyy - sum_wy*vals[0]))/sum_w, (float)MAX_VAR);
  }
  else {
    vals[0] = 0;
    vals[1] = MAX_VAR;
  }
}

__device__ void compute_dynamics(float* s, float* u, float* lwpr_input, RF_Predict* rfs1, RF_Predict* rfs2, 
				 RF_Predict* rfs3, int timestep, int numRFS1, int numRFS2, int numRFS3) 
{
  float dt = 1.0/(1.0*HZ);
  //------Problem Specific------------
  float vals[2];
  //Compute first prediction
  lwpr_input[0] = s[0]/norm_in1_d[0];
  lwpr_input[1] = s[1]/norm_in1_d[1];
  lwpr_input[2] = s[2]/norm_in1_d[2];
  lwpr_input[3] = s[3]/norm_in1_d[3];
  lwpr_input[4] = s[4]/norm_in1_d[4];
  compute_predict_conf(rfs1, lwpr_input, numRFS1, vals);
  s[0] += dt*(vals[0] + vals[1]*dm_d[T*DERIV_STATE_DIM*threadIdx.y + DERIV_STATE_DIM*timestep]);
  //Compute second prediction
  lwpr_input[0] = s[0]/norm_in2_d[0];
  lwpr_input[1] = s[1]/norm_in2_d[1];
  lwpr_input[2] = s[2]/norm_in2_d[2];
  lwpr_input[3] = s[3]/norm_in2_d[3];
  lwpr_input[4] = s[4]/norm_in2_d[4];
  compute_predict_conf(rfs2, lwpr_input, numRFS2, vals);
  s[1] += dt*(vals[0] + vals[1]*dm_d[T*DERIV_STATE_DIM*threadIdx.y + DERIV_STATE_DIM*timestep + 1]);
  //Compute third prediction
  lwpr_input[0] = s[0]/norm_in3_d[0];
  lwpr_input[1] = s[1]/norm_in3_d[1];
  lwpr_input[2] = s[2]/norm_in3_d[2];
  lwpr_input[3] = s[3]/norm_in3_d[3];
  lwpr_input[4] = s[4]/norm_in3_d[4];
  compute_predict_conf(rfs3, lwpr_input, numRFS3, vals);
  s[2] += dt*(vals[0] + vals[1]*dm_d[T*DERIV_STATE_DIM*threadIdx.y + DERIV_STATE_DIM*timestep + 2]);
  //Low pass filter controls
  s[3] += dt*(u[0] - s[3]);
  s[4] += dt*(u[1] - s[4]);
  //-----End Problem Specific----------
}

__device__ float compute_cost(float* s, float* u, float* goal)
{
  float d1 = (s[0] - goal[0]);
  float d2 = (s[1] - goal[1]);
  return d1*d1 + d2*d2;
}

__global__ void rollout_kernel(float* aug_state_costs_d, float* state_d, float* goal_d, RF_Predict* rfs1,
			       RF_Predict* rfs2, RF_Predict* rfs3, float* du_d, float* vars_d, 
			       int numRFS1, int numRFS2, int numRFS3)
{
  int tdx = threadIdx.x;
  int tdy = threadIdx.y;
  int bdx = blockIdx.x;
  if (blockDim.x*bdx+tdx < K) {
    //Initialize the local state
    float s[STATE_DIM];
    float u[CONTROL_DIM];
    float lwpr_input[N];
    float vars[CONTROL_DIM];
    int i,j;
    //Load the initial state
    for (i = 0; i < STATE_DIM; i++) {
      s[i] = state_d[i];
    }
    //Load vars
    for (i = 0; i < CONTROL_DIM; i++) {
      vars[i] = vars_d[i];
    }
    for (i = 0; i < T; i++) {
      //Start the main program loop
      for (j = 0; j < CONTROL_DIM; j++) {
	if (bdx == 0 && tdx == 0) {
	  u[j] = U_d[i*CONTROL_DIM + j];
	}
	else {
	  u[j] = U_d[i*CONTROL_DIM + j] + du_d[CONTROL_DIM*T*(blockDim.x*bdx + tdx) + i*CONTROL_DIM + j]*vars[j];
	}
      }
      //Check to see if the control commands are allowable
      //-------Problem Specific Values---------------
      if (u[0] > 1.0) {
	u[0] = 1.0;
      }
      else if (u[0] < -1.0) {
	u[0] = -1.0;
      }
      if (u[1] > 1.0) {
	u[1] = 1.0;
      }
      else if (u[1] < -1.0) {
	u[1] = -1.0;
      }
      //------End Problem Specific Values-------------
      compute_dynamics(s, u, lwpr_input, rfs1, rfs2, rfs3, i, numRFS1, numRFS2, numRFS3);
      float inst_cost = compute_cost(s,u,goal_d);
      aug_state_costs_d[M*T*((blockDim.x)*bdx + tdx) + T*tdy + i] = inst_cost;
    }
  }	
}

__global__ void expec_costs_kernel(float* state_costs_d, float* aug_state_costs_d)
{
  int tdx = threadIdx.x;
  int bdx = blockIdx.x;
  float expec_cost = 0;
  int i;
  if (tdx < T && bdx < K) {
    for (i = 0; i < M; i++) {
      expec_cost += aug_state_costs_d[M*T*bdx + T*i + tdx];
    }
    state_costs_d[T*bdx + tdx] = expec_cost/(1.0*M);
  }
}

__global__ void norm_exp_costs_kernel(float* state_costs_d)
{
  int tdx = threadIdx.x;
  int bdx = blockIdx.x;
  int index = blockDim.x*bdx + tdx;
  if (index < K) {
    float cost2go = 0;
    float nf_normal = 0;
    int i;
    for (i = T-1; i >= 0; i--) {
      cost2go += state_costs_d[T*index + i];
      nf_normal += state_costs_d[i];
      state_costs_d[T*index + i] = __expf(-10.0*cost2go/nf_normal);
    }
  }
}

//=========================================================================================
//--------------------------------END CUDA------------------------------------------------
//========================================================================================

void compute_control(float* state, float* U, float* goal, LWPR_Model model1, LWPR_Model model2,
		     LWPR_Model model3, float* vars, curandGenerator_t gen) {
  //Timing Code
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //First we create du_d, perturbations of U which reside in device memory.
  float* du_d;
  HANDLE_ERROR( cudaMalloc((void**)&du_d, K*T*CONTROL_DIM*sizeof(float)));
  curandGenerateNormal(gen, du_d, K*T*CONTROL_DIM, 0.0, 1.0);
  //Next we create dm_d perturbations of the LWPR model in device memory
  float* dm_temp;
  HANDLE_ERROR( cudaMalloc((void**)&dm_temp, M*T*DERIV_STATE_DIM*sizeof(float)));
  curandGenerateNormal(gen, dm_temp, M*T*DERIV_STATE_DIM, 0.0, 1.0);
  HANDLE_ERROR( cudaMemcpyToSymbol(dm_d, dm_temp, M*T*DERIV_STATE_DIM*sizeof(float), 0, cudaMemcpyDeviceToDevice));
  //cudaFree(dm_temp);
  //Create pointers for state, U, goal, rfs1, rfs2, and vars in device memory
  float* state_d;
  float* goal_d;
  float* vars_d;
  //Transfer relevant data from host LWPR model to device LWPR Receptive Field
  int i,j;
  RF_Predict* rfs1;
  RF_Predict* rfs2;
  RF_Predict* rfs3;
  rfs1 = (RF_Predict*)malloc(model1.sub[0].numRFS*sizeof(RF_Predict));
  rfs2 = (RF_Predict*)malloc(model2.sub[0].numRFS*sizeof(RF_Predict));
  rfs3 = (RF_Predict*)malloc(model3.sub[0].numRFS*sizeof(RF_Predict)); 
  for (i = 0; i < model1.sub[0].numRFS; i++) {
    rfTransfer(model1.sub[0].rf[i], &rfs1[i]);
  }
  for (i = 0; i < model2.sub[0].numRFS; i++) {
    rfTransfer(model2.sub[0].rf[i], &rfs2[i]);
  }
  for (i = 0; i < model3.sub[0].numRFS; i++) {
    rfTransfer(model3.sub[0].rf[i], &rfs3[i]);
  }
  //Transfer norms to float arrays
  float norm_in1[N];
  float norm_in2[N];
  float norm_in3[N];
  for (i = 0; i < N; i++) {
    norm_in1[i] = model1.norm_in[i];
  }
  for (i = 0; i < N; i++) {
    norm_in2[i] = model2.norm_in[i];
  }
  for (i = 0; i < N; i++) {
    norm_in3[i] = model3.norm_in[i];
  }
  //Create device pointers for rfs1, rfs2, norm_in1, and norm_in2
  RF_Predict* rfs1_d;
  RF_Predict* rfs2_d;
  RF_Predict* rfs3_d;
  //Allocate space for state, U, goal, rfs1, rfs2, and vars in device memory
  HANDLE_ERROR( cudaMalloc((void**)&state_d, STATE_DIM*sizeof(float)));
  HANDLE_ERROR( cudaMalloc((void**)&goal_d, STATE_DIM*sizeof(float)));
  HANDLE_ERROR( cudaMalloc((void**)&vars_d, CONTROL_DIM*sizeof(float)));
  HANDLE_ERROR( cudaMalloc((void**)&rfs1_d, model1.sub[0].numRFS*sizeof(RF_Predict)));
  HANDLE_ERROR( cudaMalloc((void**)&rfs2_d, model2.sub[0].numRFS*sizeof(RF_Predict)));
  HANDLE_ERROR( cudaMalloc((void**)&rfs3_d, model3.sub[0].numRFS*sizeof(RF_Predict)));
  //Copy state, U, goal, model1, and model2 into device memory
  HANDLE_ERROR( cudaMemcpy(state_d, state, STATE_DIM*sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR( cudaMemcpyToSymbol(U_d, U, CONTROL_DIM*T*sizeof(float), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR( cudaMemcpy(goal_d, goal, STATE_DIM*sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR( cudaMemcpy(vars_d, vars, CONTROL_DIM*sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR( cudaMemcpy(rfs1_d, rfs1, model1.sub[0].numRFS*sizeof(RF_Predict), cudaMemcpyHostToDevice));
  HANDLE_ERROR( cudaMemcpy(rfs2_d, rfs2, model2.sub[0].numRFS*sizeof(RF_Predict), cudaMemcpyHostToDevice));
  HANDLE_ERROR( cudaMemcpy(rfs3_d, rfs3, model3.sub[0].numRFS*sizeof(RF_Predict), cudaMemcpyHostToDevice));
  HANDLE_ERROR( cudaMemcpyToSymbol(norm_in1_d, norm_in1, N*sizeof(float), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR( cudaMemcpyToSymbol(norm_in2_d, norm_in2, N*sizeof(float), 0, cudaMemcpyHostToDevice));
  HANDLE_ERROR( cudaMemcpyToSymbol(norm_in3_d, norm_in3, N*sizeof(float), 0, cudaMemcpyHostToDevice));
  //Allocate space for the state costs and new controls
  //For the raw state costs
  float* aug_state_costs_d;
  HANDLE_ERROR( cudaMalloc((void**)&aug_state_costs_d, T*K*M*sizeof(float)));
  //For the averaged state costs
  float* state_costs_d;
  //For controls we just re-use du_d
  HANDLE_ERROR( cudaMalloc((void**)&state_costs_d, T*K*sizeof(float)));
  //Now we set the grid and block size
  int xBlockSize = (BLOCKSIZE-1)/M + 1;
  int yBlockSize = M;
  int xGridSize = (K-1)/xBlockSize + 1;
  dim3 dimBlock1(xBlockSize, yBlockSize, 1);
  dim3 dimGrid1(xGridSize, 1, 1);
  cudaEventRecord(start, 0);
  //Now we launch the kernel to compute the new control
  rollout_kernel<<<dimGrid1, dimBlock1>>>(aug_state_costs_d, state_d, goal_d, rfs1_d, rfs2_d, rfs3_d, du_d, vars_d, model1.sub[0].numRFS, model2.sub[0].numRFS, model3.sub[0].numRFS);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaDeviceSynchronize();
  //Wait until the kernel has finished
  dim3 dimBlock2(T, 1, 1);
  dim3 dimGrid2(K, 1, 1);
  //Compute expectation of the costs
  expec_costs_kernel<<<dimGrid2, dimBlock2>>>(state_costs_d, aug_state_costs_d);
  cudaDeviceSynchronize();
  dim3 dimBlock3(64, 1, 1);
  dim3 dimGrid3((K-1)/64 + 1, 1, 1);
  //Now we normalize the cost-to-go by the noise free path, and exponentiate by the -lambda*cost2go
  norm_exp_costs_kernel<<<dimGrid3, dimBlock3>>>(state_costs_d);
  cudaDeviceSynchronize();
  //Compute the normalizer
  //For now just do it on the CPU
  //Transfer state costs to host memory
  float* state_costs;
  state_costs = (float*)malloc(T*K*sizeof(float));
  HANDLE_ERROR( cudaMemcpy(state_costs, state_costs_d, T*K*sizeof(float), cudaMemcpyDeviceToHost));
  //Now compute the normalizer
  float* normalizer;
  normalizer = (float*)malloc(T*sizeof(float));
  for (i = 0; i < T; i++) {
    normalizer[i] = 0;
    for (j = 0; j < K; j++) {
      normalizer[i] += state_costs[T*j + i];
    }
  }
  //Compute the new controls
  //Just do on CPU for now
  //First transfer controls to host memory
  float* du;
  du = (float*)malloc(T*K*CONTROL_DIM*sizeof(float));
  HANDLE_ERROR( cudaMemcpy(du, du_d, T*K*CONTROL_DIM*sizeof(float), cudaMemcpyDeviceToHost));
  //Now compute the new control and place it in U
  float* U_new;
  U_new = (float*)malloc(T*CONTROL_DIM*sizeof(float));
  for (i = 0; i < T; i++) {
    U_new[CONTROL_DIM*i] = (state_costs[i]/normalizer[i])*U[CONTROL_DIM*i];
    U_new[CONTROL_DIM*i + 1] = (state_costs[i]/normalizer[i])*U[CONTROL_DIM*i + 1];
    for (j = 0; j < K; j++) {
      float u1 = U[CONTROL_DIM*i] + du[T*CONTROL_DIM*j + CONTROL_DIM*i]*vars[0];
      float u2 = U[CONTROL_DIM*i + 1] + du[T*CONTROL_DIM*j + CONTROL_DIM*i + 1]*vars[1];
      float u_max = 1.0;
      float u_min = -1.0;
      u1 = fmin(u1, u_max);
      u1 = fmax(u1, u_min);
      u2 = fmin(u2, u_max);
      u2 = fmax(u2, u_min);
      U_new[CONTROL_DIM*i] += (state_costs[T*j + i]/normalizer[i])*u1;
      U_new[CONTROL_DIM*i + 1] += (state_costs[T*j + i]/normalizer[i])*u2;
    }
    U[i*CONTROL_DIM] = U_new[i*CONTROL_DIM];
    U[i*CONTROL_DIM + 1] = U_new[i*CONTROL_DIM + 1];
  }
  
  //Free device arrays
  cudaFree(state_d);
  cudaFree(goal_d);
  cudaFree(rfs1_d);
  cudaFree(rfs2_d);
  cudaFree(du_d);
  cudaFree(state_costs_d);
  cudaFree(aug_state_costs_d);
  cudaFree(vars_d);
  //Free host arrays
  free(rfs1);
  free(rfs2);
  free(state_costs);
  free(du);
  free(normalizer);
  //Print timing results
  cudaEventElapsedTime(&time, start, stop); 
  printf("Kernel Time: %f ms \n", time);
}

void dynamics(float* s, float* u, float dt) {
  s[0] += dt*s[3];
  s[1] += dt*(s[4]);
  s[2] = 0;
  s[3] += dt*(u[0] - s[3]);
  s[4] += dt*(u[1] - s[4]);
}

int main() {
  LWPR_Model model1;
  LWPR_Model model2;
  LWPR_Model model3;
  char x_dot[] = {'x', '.', 'x', 'm', 'l', '\0'};
  char y_dot[] = {'y', '.', 'x', 'm', 'l', '\0'};
  char theta_dot[] = {'t', 'h', 'e', 't', 'a', '.', 'x', 'm', 'l', '\0'};
  int e1[] = {-3};
  int e2[] = {-3};
  int e3[] = {-3};
  lwpr_read_xml(&model1, x_dot, e1);
  lwpr_read_xml(&model2, y_dot, e2);
  lwpr_read_xml(&model3, theta_dot, e3); 
  float U[T*CONTROL_DIM] = {0};
  float u[CONTROL_DIM] = {0};
  float s[STATE_DIM] = {0};
  float goal[] = {10.0, -3.0, 0, 0, 0};
  float vars[] = {.30, .30};
  curandGenerator_t gen;
  float dt = (1.0)/(1.0*HZ);
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  int i,j;
  for (j = 0; j < 1; j++) {
    compute_control(s, U, goal, model1, model2, model3, vars, gen);
    u[0] = U[0];
    u[1] = U[1];
    for (i = 0; i < (T-1)*CONTROL_DIM; i++) {
      U[i] = U[i+CONTROL_DIM];
    }
    U[T-2] = 0;
    U[T-1] = 0;
    double lwpr_input[5] = {s[0], s[1], s[2], s[3], s[4]};
    double out1 = s[0];
    double out2 = s[1];
    double out3 = s[2];
    dynamics(s, u, dt);
    out1 = (s[0] - out1)/dt;
    out2 = (s[1] - out2)/dt;
    out3 = (s[2] - out3)/dt;
    lwpr_update(&model1, lwpr_input, &out1, NULL, NULL);
    lwpr_update(&model2, lwpr_input, &out2, NULL, NULL);
    lwpr_update(&model3, lwpr_input, &out3, NULL, NULL);
    printf("Current Location: (%f, %f) \n", s[0], s[1]);
  }
}
  
