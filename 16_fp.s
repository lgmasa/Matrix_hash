	.file	"16_fp.c"
	.text
	.p2align 4
	.globl	fp_init
	.type	fp_init, @function
fp_init:
.LFB87:
	.cfi_startproc
	endbr64
	movl	$0, (%rdi)
	ret
	.cfi_endproc
.LFE87:
	.size	fp_init, .-fp_init
	.p2align 4
	.globl	fp_clear
	.type	fp_clear, @function
fp_clear:
.LFB88:
	.cfi_startproc
	endbr64
	ret
	.cfi_endproc
.LFE88:
	.size	fp_clear, .-fp_clear
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"%u"
	.text
	.p2align 4
	.globl	fp_printf
	.type	fp_printf, @function
fp_printf:
.LFB89:
	.cfi_startproc
	endbr64
	movl	(%rdi), %esi
	xorl	%eax, %eax
	leaq	.LC0(%rip), %rdi
	jmp	__gmp_printf@PLT
	.cfi_endproc
.LFE89:
	.size	fp_printf, .-fp_printf
	.p2align 4
	.globl	fp_set
	.type	fp_set, @function
fp_set:
.LFB90:
	.cfi_startproc
	endbr64
	movl	(%rsi), %eax
	movl	%eax, (%rdi)
	ret
	.cfi_endproc
.LFE90:
	.size	fp_set, .-fp_set
	.p2align 4
	.globl	fp_is_equal
	.type	fp_is_equal, @function
fp_is_equal:
.LFB91:
	.cfi_startproc
	endbr64
	movl	(%rsi), %eax
	cmpl	%eax, (%rdi)
	sete	%al
	movzbl	%al, %eax
	ret
	.cfi_endproc
.LFE91:
	.size	fp_is_equal, .-fp_is_equal
	.p2align 4
	.globl	fp_is_zero
	.type	fp_is_zero, @function
fp_is_zero:
.LFB92:
	.cfi_startproc
	endbr64
	movl	(%rdi), %edx
	xorl	%eax, %eax
	testl	%edx, %edx
	sete	%al
	ret
	.cfi_endproc
.LFE92:
	.size	fp_is_zero, .-fp_is_zero
	.p2align 4
	.globl	fp_random
	.type	fp_random, @function
fp_random:
.LFB93:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	movq	%rdi, %rbx
	subq	$8, %rsp
	.cfi_def_cfa_offset 32
	movl	seeded.0(%rip), %eax
	testl	%eax, %eax
	je	.L12
.L9:
	call	rand@PLT
	movl	%eax, %ebp
	call	rand@PLT
	xorl	%edx, %edx
	sall	$16, %eax
	xorl	%ebp, %eax
	andl	$2147483647, %eax
	cmpl	$2147483647, %eax
	cmove	%edx, %eax
	movl	%eax, (%rbx)
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L12:
	.cfi_restore_state
	xorl	%edi, %edi
	call	time@PLT
	movl	%eax, %edi
	call	srand@PLT
	movl	$1, seeded.0(%rip)
	jmp	.L9
	.cfi_endproc
.LFE93:
	.size	fp_random, .-fp_random
	.p2align 4
	.globl	fp_cmp
	.type	fp_cmp, @function
fp_cmp:
.LFB94:
	.cfi_startproc
	endbr64
	movl	(%rsi), %eax
	cmpl	%eax, (%rdi)
	setne	%al
	movzbl	%al, %eax
	ret
	.cfi_endproc
.LFE94:
	.size	fp_cmp, .-fp_cmp
	.p2align 4
	.globl	fp_neg
	.type	fp_neg, @function
fp_neg:
.LFB95:
	.cfi_startproc
	endbr64
	movl	(%rsi), %eax
	movl	$2147483647, %edx
	subl	%eax, %edx
	testl	%eax, %eax
	cmovne	%edx, %eax
	movl	%eax, (%rdi)
	ret
	.cfi_endproc
.LFE95:
	.size	fp_neg, .-fp_neg
	.p2align 4
	.globl	fp_add
	.type	fp_add, @function
fp_add:
.LFB96:
	.cfi_startproc
	endbr64
	movl	(%rdx), %eax
	addl	(%rsi), %eax
	addq	$1, fp_add_count(%rip)
	leal	-2147483647(%rax), %edx
	cmpl	$2147483646, %eax
	cmova	%edx, %eax
	movl	%eax, (%rdi)
	ret
	.cfi_endproc
.LFE96:
	.size	fp_add, .-fp_add
	.p2align 4
	.globl	fp_add_plus
	.type	fp_add_plus, @function
fp_add_plus:
.LFB97:
	.cfi_startproc
	endbr64
	movl	(%rdx), %eax
	addl	(%rsi), %eax
	addq	$1, fp_add_count(%rip)
	leal	-2147483629(%rax), %edx
	cmpl	$2147483628, %eax
	cmova	%edx, %eax
	movl	%eax, (%rdi)
	ret
	.cfi_endproc
.LFE97:
	.size	fp_add_plus, .-fp_add_plus
	.p2align 4
	.globl	fp_sub
	.type	fp_sub, @function
fp_sub:
.LFB98:
	.cfi_startproc
	endbr64
	movl	(%rsi), %ecx
	movl	(%rdx), %edx
	addq	$1, fp_sub_count(%rip)
	movl	%ecx, %eax
	subl	%edx, %eax
	cmpl	%edx, %ecx
	leal	2147483647(%rax), %esi
	cmovb	%esi, %eax
	movl	%eax, (%rdi)
	ret
	.cfi_endproc
.LFE98:
	.size	fp_sub, .-fp_sub
	.p2align 4
	.globl	fp_sub_plus
	.type	fp_sub_plus, @function
fp_sub_plus:
.LFB99:
	.cfi_startproc
	endbr64
	movl	(%rsi), %ecx
	movl	(%rdx), %edx
	addq	$1, fp_sub_count(%rip)
	movl	%ecx, %eax
	subl	%edx, %eax
	cmpl	%edx, %ecx
	leal	2147483629(%rax), %esi
	cmovb	%esi, %eax
	movl	%eax, (%rdi)
	ret
	.cfi_endproc
.LFE99:
	.size	fp_sub_plus, .-fp_sub_plus
	.p2align 4
	.globl	fp_mul
	.type	fp_mul, @function
fp_mul:
.LFB100:
	.cfi_startproc
	endbr64
	movq	%rdx, %rax
	movl	(%rsi), %edx
	addq	$1, fp_mul_count(%rip)
	movl	(%rax), %eax
	imulq	%rax, %rdx
	movl	%edx, %eax
	shrq	$31, %rdx
	andl	$2147483647, %eax
	addl	%edx, %eax
	leal	-2147483647(%rax), %edx
	cmpl	$2147483646, %eax
	cmova	%edx, %eax
	movl	%eax, (%rdi)
	ret
	.cfi_endproc
.LFE100:
	.size	fp_mul, .-fp_mul
	.p2align 4
	.globl	fp_mul_plus
	.type	fp_mul_plus, @function
fp_mul_plus:
.LFB101:
	.cfi_startproc
	endbr64
	movl	(%rdx), %edx
	movl	(%rsi), %eax
	movl	$2147483629, -4(%rsp)
	movl	-4(%rsp), %ecx
	addq	$1, fp_mul_count(%rip)
	imulq	%rdx, %rax
	xorl	%edx, %edx
	divq	%rcx
	movl	%edx, (%rdi)
	ret
	.cfi_endproc
.LFE101:
	.size	fp_mul_plus, .-fp_mul_plus
	.p2align 4
	.globl	fp_pow
	.type	fp_pow, @function
fp_pow:
.LFB103:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	movq	%rdx, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	subq	$24, %rsp
	.cfi_def_cfa_offset 80
	movl	(%rsi), %r13d
	movl	$2, %esi
	movq	%rdi, 8(%rsp)
	movq	%rdx, %rdi
	call	__gmpz_sizeinbase@PLT
	testq	%rax, %rax
	je	.L35
	movq	fp_mul_count(%rip), %r15
	movq	%rax, %rbx
	movl	$1, %r14d
	xorl	%r12d, %r12d
	jmp	.L34
	.p2align 4,,10
	.p2align 3
.L31:
	imulq	%rax, %rax
	addq	$1, %r15
	movq	%r15, fp_mul_count(%rip)
	movl	%eax, %r13d
	shrq	$31, %rax
	andl	$2147483647, %r13d
	addl	%eax, %r13d
	cmpl	$2147483646, %r13d
	leal	-2147483647(%r13), %eax
	cmova	%eax, %r13d
	addq	$1, %r12
	cmpq	%r12, %rbx
	je	.L30
.L34:
	movq	%r12, %rsi
	movq	%rbp, %rdi
	call	__gmpz_tstbit@PLT
	movl	%eax, %ecx
	movl	%r13d, %eax
	testl	%ecx, %ecx
	je	.L31
	imulq	%rax, %r14
	addq	$1, %r15
	movl	%r14d, %ecx
	shrq	$31, %r14
	andl	$2147483647, %ecx
	addl	%ecx, %r14d
	leal	-2147483647(%r14), %ecx
	cmpl	$2147483646, %r14d
	cmova	%rcx, %r14
	jmp	.L31
	.p2align 4,,10
	.p2align 3
.L35:
	movl	$1, %r14d
.L30:
	movq	8(%rsp), %rax
	movl	%r14d, (%rax)
	addq	$24, %rsp
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE103:
	.size	fp_pow, .-fp_pow
	.section	.rodata.str1.1
.LC1:
	.string	"/0 is not defined"
.LC2:
	.string	"7FFFFFFD"
	.text
	.p2align 4
	.globl	fp_inv
	.type	fp_inv, @function
fp_inv:
.LFB102:
	.cfi_startproc
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	subq	$32, %rsp
	.cfi_def_cfa_offset 64
	movq	%fs:40, %rax
	movq	%rax, 24(%rsp)
	xorl	%eax, %eax
	movl	(%rsi), %eax
	testl	%eax, %eax
	je	.L47
	movq	%rsp, %r12
	movq	%rdi, %rbp
	movq	%rsi, %rbx
	movl	$16, %edx
	leaq	.LC2(%rip), %rsi
	movq	%r12, %rdi
	call	__gmpz_init_set_str@PLT
	movq	%rbp, %rdi
	movq	%r12, %rdx
	movq	%rbx, %rsi
	call	fp_pow
	movq	%r12, %rdi
	call	__gmpz_clear@PLT
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L46
	addq	$32, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L47:
	.cfi_restore_state
	movq	24(%rsp), %rax
	subq	%fs:40, %rax
	jne	.L46
	addq	$32, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	leaq	.LC1(%rip), %rdi
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	jmp	puts@PLT
.L46:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE102:
	.size	fp_inv, .-fp_inv
	.section	.rodata.str1.1
.LC3:
	.string	"3FFFFFFF"
	.text
	.p2align 4
	.globl	fp_legendre
	.type	fp_legendre, @function
fp_legendre:
.LFB104:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	.cfi_offset 3, -24
	subq	$56, %rsp
	.cfi_def_cfa_offset 80
	movl	(%rdi), %edx
	movq	%fs:40, %rax
	movq	%rax, 40(%rsp)
	xorl	%eax, %eax
	testl	%edx, %edx
	je	.L48
	leaq	16(%rsp), %rbp
	movq	%rdi, %rbx
	movl	$16, %edx
	leaq	.LC3(%rip), %rsi
	movq	%rbp, %rdi
	call	__gmpz_init_set_str@PLT
	leaq	12(%rsp), %rdi
	movq	%rbp, %rdx
	movq	%rbx, %rsi
	call	fp_pow
	movq	%rbp, %rdi
	call	__gmpz_clear@PLT
	xorl	%eax, %eax
	cmpl	$1, 12(%rsp)
	sete	%al
	leal	-1(%rax,%rax), %eax
.L48:
	movq	40(%rsp), %rdx
	subq	%fs:40, %rdx
	jne	.L55
	addq	$56, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	ret
.L55:
	.cfi_restore_state
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE104:
	.size	fp_legendre, .-fp_legendre
	.p2align 4
	.globl	fp_sqrt
	.type	fp_sqrt, @function
fp_sqrt:
.LFB105:
	.cfi_startproc
	endbr64
	pushq	%r12
	.cfi_def_cfa_offset 16
	.cfi_offset 12, -16
	pushq	%rbp
	.cfi_def_cfa_offset 24
	.cfi_offset 6, -24
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset 3, -32
	subq	$48, %rsp
	.cfi_def_cfa_offset 80
	movl	(%rsi), %edx
	movq	%fs:40, %rax
	movq	%rax, 40(%rsp)
	xorl	%eax, %eax
	testl	%edx, %edx
	jne	.L57
.L61:
	movl	$0, 0(%rbp)
.L58:
	movl	$1, %eax
.L56:
	movq	40(%rsp), %rdx
	subq	%fs:40, %rdx
	jne	.L64
	addq	$48, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%rbp
	.cfi_def_cfa_offset 16
	popq	%r12
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L57:
	.cfi_restore_state
	leaq	16(%rsp), %r12
	movq	%rsi, %rbx
	movl	$16, %edx
	leaq	.LC3(%rip), %rsi
	movq	%r12, %rdi
	call	__gmpz_init_set_str@PLT
	leaq	12(%rsp), %rdi
	movq	%r12, %rdx
	movq	%rbx, %rsi
	call	fp_pow
	movq	%r12, %rdi
	call	__gmpz_clear@PLT
	cmpl	$1, 12(%rsp)
	je	.L65
	movl	(%rbx), %eax
	testl	%eax, %eax
	je	.L61
	xorl	%eax, %eax
	jmp	.L56
	.p2align 4,,10
	.p2align 3
.L65:
	movq	%r12, %rdi
	call	__gmpz_init@PLT
	movq	%r12, %rdi
	movl	$1, %esi
	call	__gmpz_set_ui@PLT
	movq	%r12, %rsi
	movq	%r12, %rdi
	movl	$29, %edx
	call	__gmpz_mul_2exp@PLT
	movq	%rbp, %rdi
	movq	%r12, %rdx
	movq	%rbx, %rsi
	call	fp_pow
	movq	%r12, %rdi
	call	__gmpz_clear@PLT
	jmp	.L58
.L64:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE105:
	.size	fp_sqrt, .-fp_sqrt
	.local	seeded.0
	.comm	seeded.0,4,4
	.globl	fp_sub_count
	.bss
	.align 8
	.type	fp_sub_count, @object
	.size	fp_sub_count, 8
fp_sub_count:
	.zero	8
	.globl	fp_add_count
	.align 8
	.type	fp_add_count, @object
	.size	fp_add_count, 8
fp_add_count:
	.zero	8
	.globl	fp_mul_count
	.align 8
	.type	fp_mul_count, @object
	.size	fp_mul_count, 8
fp_mul_count:
	.zero	8
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
