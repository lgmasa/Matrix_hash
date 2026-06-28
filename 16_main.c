#include "16_header.h"

int main(void){
    uint32_t p_mer = P_MERSENNE;
    uint32_t p_pri = P_PRIME;
    printf("p_mer : %u\n",p_mer);
    // printf("p_pri : %u\n",p_pri);
    printf("α : "); fp4_printf(&alpha);

    state_t MDS; //MDS行列
    state_init(&MDS);

    affine16_t AFF;
    affine16_init(&AFF);
    affine16_set_A(&AFF);
    affine16_set_b(&AFF);
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

    //MDS行列をセット
    fp_set_ui(&MDS.m[0][0], 1);
    fp_set_ui(&MDS.m[0][1], 1);
    fp_set_ui(&MDS.m[0][2], 2);
    fp_set_ui(&MDS.m[0][3], 8);

    fp_set_ui(&MDS.m[1][0], 8);
    fp_set_ui(&MDS.m[1][1], 1);
    fp_set_ui(&MDS.m[1][2], 1);
    fp_set_ui(&MDS.m[1][3], 2);

    fp_set_ui(&MDS.m[2][0], 2);
    fp_set_ui(&MDS.m[2][1], 8);
    fp_set_ui(&MDS.m[2][2], 1);
    fp_set_ui(&MDS.m[2][3], 1);

    fp_set_ui(&MDS.m[3][0], 1);
    fp_set_ui(&MDS.m[3][1], 2);
    fp_set_ui(&MDS.m[3][2], 8);
    fp_set_ui(&MDS.m[3][3], 1);

    // affine16_set(&AFF);

    uint8_t block[MATRIX_STATE_BYTES] = {0};
    uint8_t digest[32];
    uint8_t msg[] = "Hello"; //文字列リテラル "Hello World" の各文字コードが uint8_t 配列に格納される
    size_t msg_len = sizeof(msg) -1; //末尾のヌル文字\0も数えてしまい、1文字多くなってしまうから1を引く

    printf("MDS :\n");
    state_print(&MDS);

    printf("msg :\n");
    printf("%s\n",msg);
    // print_bytes_hex(msg, 11);

    //throughput_matrix_hash(msg, msg_len, 100, &MDS, &AFF);

    matrix_hash(digest, MATRIX_DIGEST_BYTES, msg, msg_len, MATRIX_ROUNDS, &MDS, &AFF);

    printf("digest :\n");
    print_bytes_hex(digest, MATRIX_DIGEST_BYTES);

    state_t S;
    state_init(&S);
    // state_set_zero(&S);

    fp_set_ui(&S.m[0][0], 0);
    fp_set_ui(&S.m[0][1], P_MERSENNE - 16);
    fp_set_ui(&S.m[0][2], P_MERSENNE - 32);
    fp_set_ui(&S.m[0][3], P_MERSENNE - 48);

    fp_set_ui(&S.m[1][0], 0);
    fp_set_ui(&S.m[1][1], 0);
    fp_set_ui(&S.m[1][2], 0);
    fp_set_ui(&S.m[1][3], 0);

    fp_set_ui(&S.m[2][0], 0);
    fp_set_ui(&S.m[2][1], 0);
    fp_set_ui(&S.m[2][2], 0);
    fp_set_ui(&S.m[2][3], 0);

    fp_set_ui(&S.m[3][0], 0);
    fp_set_ui(&S.m[3][1], 0);
    fp_set_ui(&S.m[3][2], 0);
    fp_set_ui(&S.m[3][3], 0);

    printf("S :\n");
    state_print(&S);

    test_diffusion_P(&S, 0, 3, 1, &MDS, &AFF, MATRIX_ROUNDS);

    state_clear(&MDS);
    affine16_clear(&AFF);

    return 0;
}

    // fp4_t A, B, C, D, E, F, One, Zero;
    // fp16_t A16, B16, C16, D16, E16, One16, Beta16, Alpha16, inv16, res16, A16_saved;
    // fp4_init(&A); fp4_init(&B); fp4_init(&C);
    // fp4_init(&D); fp4_init(&E); fp4_init(&F);
    // fp4_init(&One); fp4_init(&Zero);
    // fp16_init(&A16); fp16_init(&B16); fp16_init(&C16);
    // fp16_init(&D16); fp16_init(&E16); fp16_init(&One16);
    // fp16_init(&Beta16); fp16_init(&Alpha16); fp16_init(&inv16); fp16_init(&res16);
    // fp16_init(&A16_saved);

    // // テストデータをランダムに生成
    // fp4_random(&A);
    // fp4_random(&B);
    // fp4_random(&C);
    // fp16_random(&A16);
    // fp16_random(&B16);
    // fp16_random(&C16);

    // printf("--- Variables ---\n");
    // printf("A = "); fp4_printf(&A);
    // printf("B = "); fp4_printf(&B);

    // // 1. 加減算のテスト: (A + B) - B == A
    // printf("\n[Test 1] Addition & Subtraction: (A + B) - B == A ... ");
    // fp4_add(&D, &A, &B); // D = A + B
    // fp4_sub(&E, &D, &B); // E = D - B
    // if (fp4_is_equal(&A, &E)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    //     printf("Expected: "); fp4_printf(&A);
    //     printf("Result:   "); fp4_printf(&E);
    // }

    // // 2. 乗算の可換性: A * B == B * A
    // printf("[Test 2] Commutativity: A * B == B * A ... ");
    // fp4_mul(&D, &A, &B); // D = A * B
    // fp4_mul(&E, &B, &A); // E = B * A
    // if (fp4_is_equal(&D, &E)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    // }

    // // 3. 分配法則: A * (B + C) == A * B + A * C
    // printf("[Test 3] Distributivity: A * (B + C) == AB + AC ... ");
    // fp4_add(&D, &B, &C); // D = B + C
    // fp4_mul(&E, &A, &D); // E = A * (B + C)

    // fp4_mul(&D, &A, &B); // D = AB
    // fp4_mul(&F, &A, &C); // F = AC
    // fp4_add(&F, &D, &F); // F = AB + AC

    // if (fp4_is_equal(&E, &F)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    // }

    // // 4. 自乗の整合性: sqr(A) == mul(A, A)
    // printf("[Test 4] Squaring: A^2 == A * A ... ");
    // fp4_sqr(&D, &A);     // D = A^2 (sqr)
    // fp4_mul(&E, &A, &A); // E = A * A (mul)
    // if (fp4_is_equal(&D, &E)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    // }

    // // 5. 単位元のテスト: A * 1 == A
    // printf("[Test 5] Identity: A * 1 == A ... ");
    // fp4_set_ui(&One, 1); // One = 1
    // fp4_mul(&D, &A, &One);
    // if (fp4_is_equal(&D, &A)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    //     printf("1 is represented as: "); fp4_printf(&One);
    //     printf("A * 1 = "); fp4_printf(&D);
    // }

    // // 6. Frobenius写像の巡回性: A^(p^4) == A
    // // 正規基底では p乗 は巡回シフトなので、4回やると元に戻るはず
    // printf("[test6] Frobenius Cycle: A^(p^4) == A ... ");
    // fp4_set(&D, &A);
    // for(int i=0; i<4; i++){
    //     fp4_frobenius_map(&D, &D); // D = D^p
    //     // printf("  Apply %d: ", i+1); fp4_printf(&D); // デバッグ用
    // }
    // if (fp4_is_equal(&D, &A)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    //     printf("Original: "); fp4_printf(&A);
    //     printf("Result:   "); fp4_printf(&D);
    // }

    // // 7. 逆元のテスト: A * A^(-1) == 1
    // printf("[Test 7] Inversion: A * A^(-1) == 1 ... ");
    // fp4_inv(&D, &A);     // D = A^(-1)
    // fp4_mul(&E, &A, &D); // E = A * A^(-1)
    // fp4_set_ui(&One, 1); // 1
    
    // if (fp4_is_equal(&E, &One)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    //     printf("Result: "); fp4_printf(&E);
    // }

    // // 8. 位数チェック: ord(A) と A^ord == 1
    // printf("[Test 8] fp4 multiplicative order of A ... ");
    // if (!fp_is_zero(&A.x0) || !fp_is_zero(&A.x1) || !fp_is_zero(&A.x2) || !fp_is_zero(&A.x3)) {
    //     mpz_t ord, group;
    //     mpz_init(ord); mpz_init(group);
    //     fp4_order(ord, &A);
    //     fp4_pow(&D, &A, ord);
    //     fp4_set_ui(&One, 1);
    //     int ok = fp4_is_equal(&D, &One);
    //     mpz_init_set_ui(group, P_MERSENNE);
    //     mpz_pow_ui(group, group, 4);
    //     mpz_sub_ui(group, group, 1);
    //     int divides = mpz_divisible_p(group, ord);
    //     gmp_printf("ord(A) = %Zd , p^4-1 = %Zd ... %s%s\n",
    //         ord, group,
    //         ok ? "A^ord=1 OK" : "A^ord!=1 NG",
    //         divides ? "" : " (ord not dividing p^4-1!)");
    //     mpz_clear(ord); mpz_clear(group);
    // } else {
    //     printf("A is zero, skipped.\n");
    // }

    // // 9. α（拡大用の元）の位数も表示しておく
    // printf("[Test 9] fp4 multiplicative order of alpha ... ");
    // mpz_t ord_alpha, two_pow, r, group_alpha;
    // mpz_inits(ord_alpha, two_pow, r, group_alpha, NULL);
    // fp4_order(ord_alpha, &alpha);
    // mpz_set_ui(group_alpha, P_MERSENNE);
    // mpz_pow_ui(group_alpha, group_alpha, 4);
    // mpz_sub_ui(group_alpha, group_alpha, 1);
    // int divides_alpha = mpz_divisible_p(group_alpha, ord_alpha);
    // gmp_printf("ord(alpha) = %Zd , p^4-1 = %Zd%s\n",
    //     ord_alpha, group_alpha,
    //     divides_alpha ? "" : " (ord not dividing p^4-1!)");
    // // 参考: 2^33で割った商と余り
    // mpz_set_ui(two_pow, 2);
    // mpz_pow_ui(two_pow, two_pow, 33);
    // mpz_tdiv_qr(r, two_pow, ord_alpha, two_pow); // r=ord/2^33, two_pow=余り
    // gmp_printf("ord(alpha)/2^33 = %Zd, rem = %Zd\n", r, two_pow);
    // mpz_clears(ord_alpha, two_pow, r, group_alpha, NULL);

    // // 10. スカラーのべき乗結果がスカラーかチェック
    // printf("[Test 10] Scalar pow keeps scalar (fp4) ... ");
    // mpz_t scalar_exp4;
    // mpz_init_set_ui(scalar_exp4, 12345);
    // fp4_set_ui(&A, 7); // スカラーをセット
    // fp4_pow(&D, &A, scalar_exp4);
    // if (fp4_is_scalar(&D)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    //     printf("Result: "); fp4_printf(&D);
    // }
    // mpz_clear(scalar_exp4);

    // // //8. 4乗非剰余を探索しつつ x^4 - alpha の既約性も表示
    // // printf("[Test 8] Search 4th-nonresidue (100 trials) with irreducibility check...\n");
    // // fp4_quartic_residue_scan(100);

    // // --- fp16 tests (拡大体 Fp^16 = Fp^4[β]/(β^4-α)) ---
    // printf("\n--- fp16 Variables ---\n");
    // printf("A16 = "); fp16_printf(&A16);
    // printf("B16 = "); fp16_printf(&B16);

    // // 1. 加減算: (A+B)-B == A
    // printf("[fp16 Test 1] Addition & Subtraction ... ");
    // fp16_add(&D16, &A16, &B16);
    // fp16_sub(&E16, &D16, &B16);
    // if (fp16_is_equal(&A16, &E16)) printf("OK\n"); else printf("NG\n");

    // // 2. 乗算の可換性
    // printf("[fp16 Test 2] Commutativity ... ");
    // fp16_mul(&D16, &A16, &B16);
    // fp16_mul(&E16, &B16, &A16);
    // if (fp16_is_equal(&D16, &E16)) printf("OK\n"); else printf("NG\n");

    // // 3. 分配法則
    // printf("[fp16 Test 3] Distributivity ... ");
    // fp16_add(&D16, &B16, &C16);         // D16 = B+C
    // fp16_mul(&E16, &A16, &D16);         // E16 = A*(B+C)
    // fp16_mul(&D16, &A16, &B16);         // D16 = AB
    // fp16_mul(&Beta16, &A16, &C16);      // reuse Beta16 as tmp = AC
    // fp16_add(&Beta16, &D16, &Beta16);   // Beta16 = AB + AC
    // if (fp16_is_equal(&E16, &Beta16)) printf("OK\n"); else printf("NG\n");

    // // 4. 自乗
    // printf("[fp16 Test 4] Squaring ... ");
    // fp16_sqr(&D16, &A16);
    // fp16_mul(&E16, &A16, &A16);
    // if (fp16_is_equal(&D16, &E16)) printf("OK\n"); else printf("NG\n");

    // // 5. 単位元: (1,0,0,0)
    // fp4_set_ui(&One, 1);
    // fp4_set_ui(&One16.x1, 0); fp4_set_ui(&One16.x2, 0); fp4_set_ui(&One16.x3, 0);
    // fp4_set(&One16.x0, &One);
    // printf("[fp16 Test 5] Identity ... ");
    // fp16_mul(&D16, &A16, &One16);
    // if (fp16_is_equal(&D16, &A16)) printf("OK\n"); else { printf("NG\n"); printf("1 = "); fp16_printf(&One16); }

    // // 6. β^4 = α のチェック (β = (0,1,0,0))
    // fp4_set_ui(&Beta16.x0, 0); fp4_set_ui(&Beta16.x2, 0); fp4_set_ui(&Beta16.x3, 0);
    // fp4_set(&Beta16.x1, &One); // β の係数は1
    // fp4_set_ui(&Alpha16.x1, 0); fp4_set_ui(&Alpha16.x2, 0); fp4_set_ui(&Alpha16.x3, 0);
    // fp4_set(&Alpha16.x0, &alpha); // α を定数項に
    // fp16_set(&D16, &Beta16);
    // for(int i=0;i<3;i++) fp16_mul(&D16, &D16, &Beta16); // D16 = β^4
    // printf("[fp16 Test 6] beta^4 == alpha ... ");
    // if (fp16_is_equal(&D16, &Alpha16)) printf("OK\n"); else { printf("NG\n"); printf("beta^4 = "); fp16_printf(&D16); }

    // // 7. スカラーのべき乗結果がスカラーかチェック
    // printf("[fp16 Test 7] Scalar pow keeps scalar (fp16) ... ");
    // mpz_t scalar_exp16;
    // mpz_init_set_ui(scalar_exp16, 12345);
    // fp16_set(&A16_saved, &A16); // 元のランダム値を保存
    // fp4_set_ui(&A16.x0, 5); // Fp4 スカラー
    // fp4_set_ui(&A16.x1, 0); fp4_set_ui(&A16.x2, 0); fp4_set_ui(&A16.x3, 0);
    // fp16_pow(&D16, &A16, scalar_exp16);
    // if (fp16_is_scalar(&D16)) {
    //     printf("OK\n");
    // } else {
    //     printf("NG\n");
    //     printf("Result: "); fp16_printf(&D16);
    // }
    // mpz_clear(scalar_exp16);
    // fp16_set(&A16, &A16_saved); // 以降のテスト用に元に戻す

    // fp16_inv(&inv16,&A16);
    // printf("1/A16 = "); fp16_printf(&inv16);

    // fp16_mul(&res16,&A16,&inv16);

    // printf("A16*1/A16 = "); fp16_printf(&res16);


    // int iters = 100;
    // bench_fp_add_avg(iters, 1000000);
    // bench_fp_sub_avg(iters, 1000000);
    // bench_fp_mul_avg(iters, 1000000);
    // bench_fp_add_plus_avg(iters, 1000000);
    // bench_fp_sub_plus_avg(iters, 1000000);
    // bench_fp_mul_plus_avg(iters, 1000000);

    // //--- Benchmark fp4_mul ---
    // bench_fp4_mul_avg(iters, 1000000);

    // // --- Benchmark fp4_mul_new ---
    // bench_fp4_mul_new_avg(iters, 1000000);

    // // --- Benchmark fp4_mul_karatsuba ---
    // bench_fp4_mul_karatsuba_avg(iters, 1000000);

    // // --- Benchmark fp4_mul_slow ---
    // bench_fp4_mul_slow_avg(iters, 1000000);

    // // --- Benchmark fp16_inv ---
    // int inv_iters = 100;
    // bench_fp16_inv_avg(inv_iters, 1000000);

    // // --- Benchmark fp16_inv_new ---
    // bench_fp16_inv_new_avg(inv_iters, 1000000);

    // // --- Benchmark fp16_inv_karatsuba ---
    // bench_fp16_inv_karatsuba_avg(inv_iters, 1000000);

    // // --- Benchmark fp16_inv_slow ---
    // bench_fp16_inv_slow_avg(inv_iters, 1000000);


