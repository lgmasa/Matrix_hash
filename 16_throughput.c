#include "16_header.h"

double now_sec(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void benchmark_matrix_hash(size_t msg_len, int iterations, const state_t *MDS, const affine16_t *AFF){
    uint8_t *msg = malloc(msg_len);
    uint8_t digest[MATRIX_DIGEST_BYTES];

    if (msg == NULL) {
        printf("malloc failed\n");
        return;
    }

    /*
      入力メッセージを適当に初期化
    */
    for (size_t i = 0; i < msg_len; i++) {
        msg[i] = (uint8_t)(i & 0xff);
    }

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

    double start = now_sec();

    /*
      最適化で消されないように digest の一部を使う
    */
    volatile uint8_t sink = 0;

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

    double elapsed = end - start;

    uint64_t total_bits = (uint64_t)msg_len * 8ULL * (uint64_t)iterations;
    double throughput_bps = (double)total_bits / elapsed;
    double throughput_mbps = throughput_bps / 1000000.0;

    printf("message length : %zu bytes\n", msg_len);
    printf("iterations     : %d\n", iterations);
    printf("elapsed time   : %.6f sec\n", elapsed);
    printf("throughput     : %.3f bit/s\n", throughput_bps);
    printf("throughput     : %.3f Mbps\n", throughput_mbps);
    printf("sink           : %u\n", sink);

    free(msg);
}