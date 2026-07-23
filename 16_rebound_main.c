#include "16_header.h"
void run_rebound_experiments(const state_t *MDS, const affine16_t *AFF);
int main(void){
    state_t MDS; state_init(&MDS);
    affine16_t AFF; affine16_init(&AFF);
    affine16_set_A(&AFF); affine16_set_b(&AFF);
    // MDS = circ(1,1,2,8) （リポジトリと同じ）
    uint32_t c[4] = {1,1,2,8};
    for(int r=0;r<4;r++) for(int j=0;j<4;j++) fp_set_ui(&MDS.m[r][j], c[(j-r+4)%4]);
    run_rebound_experiments(&MDS, &AFF);
    return 0;
}
