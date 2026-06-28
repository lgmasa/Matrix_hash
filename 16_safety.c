#include "16_header.h"

//差分・拡散性評価
//入力の一部を変化させたとき、その影響が状態全体や最終出力に十分広がるかを確認する

//2つの行列の各成分を比較し、要素が異なる箇所の数を出力する
int state_count_diff_components(const state_t *A, const state_t *B){
    int count = 0;

    for(int i = 0; i < 4; i++){
        for(int j = 0; j< 4; j++){
            if(fp_is_equal(&A->m[i][j], &B->m[i][j]) == 0){
            count ++;
            }
        }
    }
    return count;
}

//2つの行列間で値が異なる箇所を1、同じ箇所を0として出力する
void state_print_diff_map(const state_t *A, const state_t *B){
    printf("diff map:\n");

    for(int i = 0; i < 4; i++){
        for(int j = 0; j< 4; j++){
            if(fp_is_equal(&A->m[i][j], &B->m[i][j]) == 0){
                printf("1 ");
            } else{
                printf("0 ");
            }
        }
        printf("\n");
    }
}

//行列Sの、diff_row行diff_col列の値に＋diff_valueしてラウンド処理を行う→Sとの差異を確認
void test_diffusion_P(const state_t *S, int diff_row, int diff_col, uint32_t diff_value, const state_t *MDS, affine16_t *AFF , int max_rounds){

    if (diff_row < 0 || diff_row >= 4 || diff_col < 0 || diff_col >= 4){
    printf("invalid diff position\n");
    return;
    }

    state_t S1,S2,T1,T2;
    fp_t d;
    state_init(&S1); state_init(&S2); state_init(&T1); state_init(&T2);
    fp_init(&d);

    state_copy(&S1, S);
    state_copy(&S2, &S1);

    fp_set_ui(&d, diff_value);
    fp_add(&S2.m[diff_row][diff_col], &S2.m[diff_row][diff_col], &d); //diff_row行diff_col列の値に＋diff_value

    printf("Initial difference:\n");
    printf("diff position = (%d, %d)\n",diff_row,diff_col);
    printf("diff_value = %u\n", diff_value);
    printf("diff components = %d / 16\n", state_count_diff_components(&S1, &S2));
    state_print_diff_map(&S1, &S2);

    for(int r = 1; r <= max_rounds; r++){
        matrix_permutation_P(&T1, &S1, r, MDS, AFF);
        matrix_permutation_P(&T2, &S2, r, MDS, AFF);

        printf("T1 :\n");
        state_print(&T1);
        printf("T2 :\n");
        state_print(&T2);

        printf("\nAfter %d rounds of P:\n",r);
        printf("diff components = %d / 16\n", state_count_diff_components(&T1, &T2));
        state_print_diff_map(&T1, &T2);
    }

    state_clear(&S1); state_clear(&S2); state_clear(&T1); state_clear(&T2);
    fp_clear(&d);
}