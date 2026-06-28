#include "16_header.h"

/*
  matrix_hash のスループット計測関数

  ここで言うスループットは「出力ハッシュ値が 1bit 出力されるのに何秒かかるか」を表す。

  測定方法:
    1. 256bit (= MATRIX_DIGEST_BYTES * 8) のハッシュ値を出力する処理を
       iterations 回（100万回程度）繰り返し、合計時間を計測する。
    2. 合計時間を iterations で割り、1回のハッシュ化にかかる平均時間を求める。
    3. その平均時間を出力ビット数(256)で割り、1bit 出力するのにかかる秒数を導出する。
*/

double now_sec(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void throughput_matrix_hash(const uint8_t *msg, size_t msg_len, int iterations, const state_t *MDS, const affine16_t *AFF){
    uint8_t digest[MATRIX_DIGEST_BYTES];

    /*
      ウォームアップ
      初回実行の影響を少し避ける
    */
    for (int i = 0; i < 10; i++) {
        matrix_hash(
            digest,
            MATRIX_DIGEST_BYTES,
            msg,
            msg_len,
            MATRIX_ROUNDS,
            MDS,
            AFF
        );
    }

    /*
      最適化で消されないように digest の一部を使う
    */
    volatile uint8_t sink = 0;

    double start = now_sec();

    for (int i = 0; i < iterations; i++) {
        matrix_hash(
            digest,
            MATRIX_DIGEST_BYTES,
            msg,
            msg_len,
            MATRIX_ROUNDS,
            MDS,
            AFF
        );

        sink ^= digest[0];
    }

    double end = now_sec();

    double elapsed = end - start;                       // 全体の経過時間 [sec]

    /*
      出力ビット数(256bit)
    */
    size_t output_bits = (size_t)MATRIX_DIGEST_BYTES * 8;

    double sec_per_hash = elapsed / (double)iterations; // 1回のハッシュ化にかかる平均時間 [sec/hash]
    double sec_per_bit  = sec_per_hash / (double)output_bits; // 1bit 出力するのにかかる時間 [sec/bit]

    /*
      1秒あたりに出力できるビット数 [bit/s]
      = 出力した総ビット数 / 経過時間  (sec_per_bit の逆数)
    */
    uint64_t total_bits = (uint64_t)output_bits * (uint64_t)iterations; // 出力した総ビット数
    double bits_per_sec = (double)total_bits / elapsed;                 // [bit/s]

    printf("message length : %zu bytes\n", msg_len);
    printf("output length  : %zu bits\n", output_bits);
    printf("iterations     : %d\n", iterations);
    printf("elapsed time   : %.6f sec\n", elapsed);
    printf("time per hash  : %.6e sec/hash\n", sec_per_hash);
    printf("throughput     : %.6e sec/bit\n", sec_per_bit);
    printf("throughput     : %.3f bit/s\n", bits_per_sec);
    printf("throughput     : %.3f Mbps\n", bits_per_sec / 1000000.0);
    printf("sink           : %u\n", sink);
}
