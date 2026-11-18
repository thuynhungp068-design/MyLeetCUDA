__global__ void merge_attn_states(
    float* output,
    const float* prefix_output,
    const float* prefix_lse,
    const float* suffix_output,
    const float* suffix_lse,
    int num_tokens,int num_heads,int head_size){
    
    constexpr int kNumThreads=256;
    constexpr int kPackSize=16/sizeof(float);
    using pack_t=uint4;

    const int threads_per_head=head_size/kPackSize;
    const int total_threads=num_tokens*num_heads*threads_per_head;

    const int global_idx=blockIdx.x*kNumThreads+threadIdx.x;
    if(global_idx>=total_threads)return;

    const int token_head_idx=global_idx/threads_per_head;
    const int pack_idx=global_idx%threads_per_head;

    const int token_idx=token_head_idx/num_heads;
    const int head_idx=token_head_idx%num_heads;

    const int pack_offset=pack_idx*kPackSize;
    const int head_offset=token_idx*num_heads*head_size+head_idx*head_size;

    const float* prefix_head=prefix_output+head_offset;
    const float* suffix_head=suffix_output+head_offset;
    float* output_head=output+head_offset;

    float p_lse=prefix_lse[head_idx*num_tokens+token_idx];
    float s_lse=suffix_lse[head_idx*num_tokens+token_idx];

    p_lse=isinf(p_lse)?-FLT_MAX:p_lse;
    s_lse=isinf(s_lse)?-FLT_MAX:s_lse;

    const float max_lse=fmaxf(p_lse,s_lse);
    p_lse-=max_lse;
    s_lse-=max_lse;

    const float p_se=expf(p_lse);
    const float s_se=expf(s_lse);
    const float out_se=p_se+s_se;
    const float p_scale=p_se/out_se;
    const float s_scale=s_se/out_se;

    if(pack_offset<head_size){
        pack_t p_pack=reinterpret_cast<const pack_t*>(prefix_head)[pack_idx];
        pack_t s_pack=reinterpret_cast<const pack_t*>(suffix_head)[pack_idx];
        pack_t out_pack;
    }

#pragma unroll
    for(int i=0;i<kPackSize;i++){
        const float p_v=reinterpret_cast<const float*>(&p_pack)[i];
        const float s_v=reinterpret_cast<const float*>(&s_pack)[i];
        const float o_v=p_v*p_scale+s_v*s_scale;
        reinterpret_cast<float*>(&out_pack)[i]=o_v;
    }

    reinterpret_cast<pack_t*>(output_head)[pack_idx]=o_pack;
}