// Copyright ©2024 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// AVX2/FMA optimized version of GemvN
// y = alpha * A * x + beta * y

//go:build !noasm && !gccgo && !safe
// +build !noasm,!gccgo,!safe

#include "textflag.h"

// func GemvNAVX2(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr)
TEXT ·GemvNAVX2(SB), NOSPLIT, $0
	MOVQ    m+0(FP), CX           // m
	MOVQ    n+8(FP), BX           // n
	MOVSD   alpha+16(FP), X15     // alpha
	MOVQ    a_base+24(FP), DI     // a
	MOVQ    lda+48(FP), R12       // lda
	MOVQ    x_base+56(FP), SI     // x
	MOVQ    incX+80(FP), R8       // incX
	MOVSD   beta+88(FP), X14      // beta
	MOVQ    y_base+96(FP), DX     // y
	MOVQ    incY+120(FP), R10     // incY
	
	// Convert increments to bytes
	SHLQ    $3, R8                // incX *= 8
	SHLQ    $3, R10               // incY *= 8
	SHLQ    $3, R12               // lda *= 8
	
	TESTQ   CX, CX
	JZ      end
	TESTQ   BX, BX
	JZ      end
	
	// For each row of A
row_loop:
	MOVQ    DI, AX                // Current row pointer
	MOVQ    SI, R11               // Reset x pointer
	MOVQ    BX, R13               // n counter
	
	VXORPD  Y0, Y0, Y0            // Clear accumulator
	
	// Check if we can use unit stride for x
	CMPQ    R8, $8
	JNE     non_unit_stride
	
	// Unit stride - process 4 elements at a time
	MOVQ    R13, R14
	SHRQ    $2, R14               // R14 = n / 4
	JZ      unit_remainder
	
unit_loop_4:
	VMOVUPD (AX), Y1              // Load 4 elements from A
	VMOVUPD (R11), Y2             // Load 4 elements from x
	VFMADD231PD Y2, Y1, Y0        // Y0 += Y1 * Y2
	
	ADDQ    $32, AX
	ADDQ    $32, R11
	DECQ    R14
	JNZ     unit_loop_4
	
unit_remainder:
	MOVQ    R13, R14
	ANDQ    $3, R14               // R14 = n % 4
	JZ      reduce
	
unit_loop_1:
	MOVSD   (AX), X1
	MOVSD   (R11), X2
	MULSD   X2, X1
	ADDSD   X1, X0                // Use lower part of Y0
	
	ADDQ    $8, AX
	ADDQ    $8, R11
	DECQ    R14
	JNZ     unit_loop_1
	JMP     reduce
	
non_unit_stride:
	// Non-unit stride - process one element at a time
non_unit_loop:
	MOVSD   (AX), X1
	MOVSD   (R11), X2
	MULSD   X2, X1
	ADDSD   X1, X0
	
	ADDQ    $8, AX
	ADDQ    R8, R11               // Add incX
	DECQ    R13
	JNZ     non_unit_loop
	
reduce:
	// Horizontal sum of Y0
	VEXTRACTF128 $1, Y0, X1       // Extract upper 128 bits
	ADDPD   X1, X0                // Add upper to lower
	MOVHLPS X0, X1                // Move high 64 bits to low of X1
	ADDSD   X1, X0                // Final sum in X0
	
	// Scale by alpha
	MULSD   X15, X0
	
	// Load y[i], scale by beta, and add
	MOVSD   (DX), X1
	MULSD   X14, X1
	ADDSD   X1, X0
	
	// Store result
	MOVSD   X0, (DX)
	
	// Advance to next row
	ADDQ    R12, DI               // a += lda
	ADDQ    R10, DX               // y += incY
	
	DECQ    CX
	JNZ     row_loop
	
end:
	VZEROUPPER
	RET
