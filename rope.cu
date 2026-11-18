__global__ void rope(float* x,float* out,int seq_len,int N){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    float x1=x[idx*2];
    float x2=x[idx*2+1];
    int token_pos=idx/N;
    int token_idx=idx%N;

    float exp_v=1.0f/powf(10000.0f,2*token_idx/(N*2.0f));
    float sin_v=sinf(token_pos*exp_v);
    float cos_v=cosf(token_pos*exp_v);

    float out1=x1*cos_v-x2*sin_v;
    float out2=x1*sin_v+x2*cos_v;
    out[idx*2]=out1;
    out[idx*2+1]=out2;
}