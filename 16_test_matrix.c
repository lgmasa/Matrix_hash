/*
 * 16_hash_test.c --- MATRIX ハッシュ全体が正しく動くかを参照値なしで確認する
 *
 *  参照 test vector が無いので「正しいハッシュが満たすべき性質」を確認する:
 *   1) 決定性       : 同じ入力 -> 常に同じ digest
 *   2) 入力感度     : 1ビット違う入力 -> 別の digest
 *   3) 長さ反映     : 長さ違いの入力 -> 別の digest
 *   4) 非退化       : digest が全0でない
 *   5) アバランシェ : 入力1ビット反転で出力ビットの約1/2が反転(理想0.5)
 *
 *  digest_len で素数が切り替わる:  8byte->2^7-1, 32byte->2^31-1, 64byte->2^127-1
 *  7/31 は対照(既に動く)、127 が本命。
 *
 * build: gcc -O2 16_hash_test.c 16_fp.c 16_fp4.c 16_fp16.c 16_matrix.c -lgmp -o 16_hash_test
 *        (16_main.c は main 重複を避けるため含めない)
 * run:   ./16_hash_test
 */
#include "16_header.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

static int popcount_bytes(const uint8_t *a, const uint8_t *b, size_t len){
    int c=0;
    for(size_t i=0;i<len;i++){
        uint8_t x = a[i]^b[i];
        while(x){ c += x&1; x >>= 1; }
    }
    return c;
}

/* 1つの素数(digest_len で選択)についてハッシュ全体を検査 */
static int test_hash_for(size_t digest_len, const state_t *MDS, const affine16_t *AFF){
    const int R = MATRIX_ROUNDS;
    uint8_t d1[MATRIX_DIGEST_BYTES_MAX], d2[MATRIX_DIGEST_BYTES_MAX];
    uint8_t msg[256], msg2[256];
    int ok = 1;

    /* この digest_len でどの素数が選ばれるか表示(matrix_hash 内部でも選ぶが確認用) */
    if(!field_select_for_output(digest_len*8)){
        printf("  digest_len=%zu は未対応\n", digest_len); return 0;
    }
    printf("== digest_len=%zu byte (%zu bit) -> FP_BITS=%u ==\n",
           digest_len, digest_len*8, FP_BITS);

    /* 1) 決定性: 同入力を2回 */
    {
        int fail=0;
        for(int t=0;t<200;t++){
            size_t mlen = 1 + (rand()%64);
            for(size_t i=0;i<mlen;i++) msg[i]=(uint8_t)rand();
            matrix_hash(d1, digest_len, msg, mlen, R, MDS, AFF);
            matrix_hash(d2, digest_len, msg, mlen, R, MDS, AFF);
            if(memcmp(d1,d2,digest_len)!=0){ fail++; }
        }
        printf("   1) 決定性(同入力->同digest)     : %s\n", fail?"NG":"OK");
        ok &= (fail==0);
    }

    /* 2) 入力感度: 1ビット違いで別digest */
    {
        int same=0, tested=0;
        for(int t=0;t<200;t++){
            size_t mlen = 1 + (rand()%64);
            for(size_t i=0;i<mlen;i++) msg[i]=(uint8_t)rand();
            memcpy(msg2,msg,mlen);
            int bit = rand()% (int)(mlen*8);
            msg2[bit/8] ^= (uint8_t)(1u << (bit%8));   /* 1ビット反転 */
            matrix_hash(d1, digest_len, msg,  mlen, R, MDS, AFF);
            matrix_hash(d2, digest_len, msg2, mlen, R, MDS, AFF);
            tested++;
            if(memcmp(d1,d2,digest_len)==0) same++;    /* 一致してしまったら異常 */
        }
        printf("   2) 入力感度(1bit違い->別digest)  : %s\n", same?"NG":"OK");
        ok &= (same==0);
    }

    /* 3) 長さ反映: 末尾に1バイト足すと別digest */
    {
        int same=0;
        for(int t=0;t<200;t++){
            size_t mlen = 1 + (rand()%63);
            for(size_t i=0;i<mlen+1;i++) msg[i]=(uint8_t)rand();
            matrix_hash(d1, digest_len, msg, mlen,   R, MDS, AFF);
            matrix_hash(d2, digest_len, msg, mlen+1, R, MDS, AFF);
            if(memcmp(d1,d2,digest_len)==0) same++;
        }
        printf("   3) 長さ反映(長さ違い->別digest)  : %s\n", same?"NG":"OK");
        ok &= (same==0);
    }

    /* 4) 非退化: digest が全0でない */
    {
        const char *m = "MATRIX";
        matrix_hash(d1, digest_len, (const uint8_t*)m, 6, R, MDS, AFF);
        int allzero=1; for(size_t i=0;i<digest_len;i++) if(d1[i]){allzero=0;break;}
        printf("   4) 非退化(digest!=全0)           : %s\n", allzero?"NG":"OK");
        ok &= (!allzero);
    }

    /* 5) アバランシェ: 1bit反転で出力の何割が反転するか(理想 0.5) */
    {
        const int N=3000;
        long total_flipped=0; long total_bits=(long)digest_len*8*N;
        for(int t=0;t<N;t++){
            size_t mlen = 1 + (rand()%64);
            for(size_t i=0;i<mlen;i++) msg[i]=(uint8_t)rand();
            memcpy(msg2,msg,mlen);
            int bit = rand()%(int)(mlen*8);
            msg2[bit/8] ^= (uint8_t)(1u << (bit%8));
            matrix_hash(d1, digest_len, msg,  mlen, R, MDS, AFF);
            matrix_hash(d2, digest_len, msg2, mlen, R, MDS, AFF);
            total_flipped += popcount_bytes(d1,d2,digest_len);
        }
        double frac = (double)total_flipped/(double)total_bits;
        int good = (frac > 0.45 && frac < 0.55);      /* 理想0.5付近 */
        printf("   5) アバランシェ(反転率, 理想0.5)  : %.4f  %s\n", frac, good?"OK":"NG");
        ok &= good;
    }

    printf("   => %s\n\n", ok?"この素数はハッシュOK":"NG あり");
    return ok;
}

int main(void){
    srand(12345);   /* 再現性 */

    /* MDS と AFF を用意(値は小さく素数非依存なので1回でよい) */
    state_t MDS; setup_MDS(&MDS);
    affine16_t AFF; affine16_init(&AFF); affine16_set_A(&AFF); affine16_set_b(&AFF);
    printf("MDS :\n");
    state_print(&MDS);
    printf("A:\n");
    affine16_print_A(&AFF);
    printf("b:\n");
    affine16_print_b(&AFF);

    int A = affine16_is_regular(&AFF);

    if (A){
        printf("A is regular\n");
    }else{
        printf("A is not regular\n");
    }

    printf("=== MATRIX ハッシュ全体テスト(参照値なし・性質ベース) ===\n\n");
    int all = 1;
    all &= test_hash_for(8,  &MDS, &AFF);   /* -> 2^7-1  対照 */
    all &= test_hash_for(32, &MDS, &AFF);   /* -> 2^31-1 対照 */
    all &= test_hash_for(64, &MDS, &AFF);   /* -> 2^127-1 本命 */
    printf("=== 総合: %s ===\n", all?"全素数 PASS":"FAIL あり(上のNG参照)");
    return all?0:1;
}