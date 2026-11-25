// Baseline and Crescent DPUs for the area / critical path synthesis (Yosys + Nangate45).
//
// A DPU computes one 16-element FP4 (E2M1) dot product per cycle:
//   16 elementwise products -> adder tree -> scale multiply -> accumulate.
// The adder tree is a word-level 3:2 carry-save (Wallace) tree with one final carry-propagate adder.
//
// Crescent DPU = baseline DPU + Correction Unit. For each operand (A, B) of the 16-element group:
//   * a 4-bit index picks the one element that carries the E2M3 extension,
//   * its 2 extra mantissa bits, shifted by that element's own exponent, give the missing value (delta),
//   * delta is multiplied by the other operand's E2M1 value at the same index.
// The two correction products are two more operands of the same carry-save tree (16 + 2 = 18).
// A 3:2 tree needs 6 levels for both 16 and 18 operands, and the correction products join after the
// first level (they arrive later, after the index select and multiply), so the critical path is not
// lengthened. The delta_a * delta_b cross term is dropped. Both terms are added before the scale
// multiply, so one scale multiply covers the main and the correction terms.
//
// Fixed point: values are 8x their real value, so every E2M1 value and every E2M3 delta quantum
// (0.125) is an integer.
//
// For Linear layers only the activation side is extended; both correction slots then read activation
// elements through a mode mux (not modeled; it reuses the same arithmetic). For QK^T both operands
// are activations.

// FP4 (E2M1) decode, 8x real value, signed.
module fp4_decode8(input [3:0] code, output signed [7:0] val);
    wire sign = code[3];
    wire [1:0] ee = code[2:1];
    wire m = code[0];
    reg [5:0] mag;
    always @* begin
        case (ee)
            2'b00: mag = m ? 6'd4  : 6'd0;
            2'b01: mag = m ? 6'd12 : 6'd8;
            2'b10: mag = m ? 6'd24 : 6'd16;
            2'b11: mag = m ? 6'd48 : 6'd32;
        endcase
    end
    assign val = sign ? -$signed({1'b0, mag}) : $signed({1'b0, mag});
endmodule

// 3:2 carry-save adder on W-bit words.
module csa #(parameter W = 20)(input [W-1:0] a, b, c, output [W-1:0] s, cy);
    assign s  = a ^ b ^ c;
    assign cy = ((a & b) | (a & c) | (b & c)) << 1;
endmodule

// 16 FP4 x FP4 products, sign-extended to W bits.
module products16 #(parameter W = 20)(
    input  [4*16-1:0] a_flat,
    input  [4*16-1:0] w_flat,
    output [16*W-1:0] prods
);
    genvar i;
    generate
        for (i = 0; i < 16; i = i + 1) begin : P
            wire signed [7:0] av, wv;
            fp4_decode8 da(.code(a_flat[i*4+3:i*4]), .val(av));
            fp4_decode8 dw(.code(w_flat[i*4+3:i*4]), .val(wv));
            wire signed [15:0] p = av * wv;
            assign prods[i*W +: W] = {{(W-16){p[15]}}, p};
        end
    endgenerate
endmodule

// 16-operand carry-save tree (baseline adder tree).
module tree16(
    input  [16*20-1:0] prods,
    output signed [19:0] sum
);
    wire [19:0] x[0:15];
    genvar i;
    generate for (i = 0; i < 16; i = i + 1) begin : X assign x[i] = prods[i*20 +: 20]; end endgenerate
    wire [19:0] s0, c0;  csa #(20) u0(.a(x[0]), .b(x[1]), .c(x[2]), .s(s0), .cy(c0));
    wire [19:0] s1, c1;  csa #(20) u1(.a(x[3]), .b(x[4]), .c(x[5]), .s(s1), .cy(c1));
    wire [19:0] s2, c2;  csa #(20) u2(.a(x[6]), .b(x[7]), .c(x[8]), .s(s2), .cy(c2));
    wire [19:0] s3, c3;  csa #(20) u3(.a(x[9]), .b(x[10]), .c(x[11]), .s(s3), .cy(c3));
    wire [19:0] s4, c4;  csa #(20) u4(.a(x[12]), .b(x[13]), .c(x[14]), .s(s4), .cy(c4));
    wire [19:0] s5, c5;  csa #(20) u5(.a(s0), .b(c0), .c(s1), .s(s5), .cy(c5));
    wire [19:0] s6, c6;  csa #(20) u6(.a(c1), .b(s2), .c(c2), .s(s6), .cy(c6));
    wire [19:0] s7, c7;  csa #(20) u7(.a(s3), .b(c3), .c(s4), .s(s7), .cy(c7));
    wire [19:0] s8, c8;  csa #(20) u8(.a(s5), .b(c5), .c(s6), .s(s8), .cy(c8));
    wire [19:0] s9, c9;  csa #(20) u9(.a(c6), .b(s7), .c(c7), .s(s9), .cy(c9));
    wire [19:0] s10, c10;  csa #(20) u10(.a(s8), .b(c8), .c(s9), .s(s10), .cy(c10));
    wire [19:0] s11, c11;  csa #(20) u11(.a(c9), .b(c4), .c(x[15]), .s(s11), .cy(c11));
    wire [19:0] s12, c12;  csa #(20) u12(.a(s10), .b(c10), .c(s11), .s(s12), .cy(c12));
    wire [19:0] s13, c13;  csa #(20) u13(.a(s12), .b(c12), .c(c11), .s(s13), .cy(c13));
    assign sum = s13 + c13;   // single carry-propagate adder
endmodule

// Main datapath: 16 products and the adder tree (runs in parallel with the Correction Unit).
module main_path(
    input  [4*16-1:0] a_flat,
    input  [4*16-1:0] w_flat,
    output signed [19:0] sum
);
    wire [16*20-1:0] prods;
    products16 #(20) PR(.a_flat(a_flat), .w_flat(w_flat), .prods(prods));
    tree16 T(.prods(prods), .sum(sum));
endmodule

// Scale multiply and accumulate (W = width of the adder tree output).
module scale_acc #(parameter W = 20)(
    input  signed [W-1:0] sum,
    input  signed [15:0] s_a,
    input  signed [15:0] s_b,
    input  signed [31:0] c_in,
    output signed [31:0] acc_out
);
    wire signed [31:0] scale = s_a * s_b;
    wire signed [W+31:0] scaled = sum * scale;
    assign acc_out = c_in + scaled[31:0];
endmodule

// Baseline DPU.
module dpe_plain(
    input  [4*16-1:0] a_flat,
    input  [4*16-1:0] w_flat,
    input  signed [15:0] s_a,
    input  signed [15:0] s_b,
    input  signed [31:0] c_in,
    output signed [31:0] acc_out
);
    wire signed [19:0] main_sum;
    main_path MP(.a_flat(a_flat), .w_flat(w_flat), .sum(main_sum));
    scale_acc #(20) SA(.sum(main_sum), .s_a(s_a), .s_b(s_b), .c_in(c_in), .acc_out(acc_out));
endmodule

// 1-of-16 multiplexer: the 4-bit code at `index`.
module sel16(
    input  [4*16-1:0] flat,
    input  [3:0] index,
    output [3:0] code
);
    assign code = flat[index*4 +: 4];
endmodule

// Value of the 2 extra mantissa bits of an E2M3-extended element, at that element's exponent (8x).
module e2m3_delta(
    input  [3:0] code,
    input  [1:0] extra_m,
    output signed [4:0] delta
);
    wire sign = code[3];
    wire [1:0] ee = code[2:1];
    // quantum(ee) = 1,1,2,4 for ee=00,01,10,11 -> shift amount 0,0,1,2
    wire [1:0] shift_amt = (ee == 2'b00) ? 2'd0 : (ee - 2'd1);
    wire [3:0] contrib_u = {2'b0, extra_m} << shift_amt;  // 0..12
    assign delta = sign ? -$signed({1'b0, contrib_u}) : $signed({1'b0, contrib_u});
endmodule

// Correction arithmetic: from the selected codes to the two correction products.
module correction_arith(
    input  [3:0] a_at_ia, w_at_ia,    // operand A's extended element and the W element at its index
    input  [3:0] a_at_ib, w_at_ib,    // operand B's extended element and the A element at its index
    input  [1:0] mant_a, mant_b,
    output signed [12:0] corr_a,
    output signed [12:0] corr_b
);
    wire signed [4:0] delta_a, delta_b;
    wire signed [7:0] w_val, a_val;
    e2m3_delta  DA(.code(a_at_ia), .extra_m(mant_a), .delta(delta_a));
    e2m3_delta  DB(.code(w_at_ib), .extra_m(mant_b), .delta(delta_b));
    fp4_decode8 PW(.code(w_at_ia), .val(w_val));
    fp4_decode8 PA(.code(a_at_ib), .val(a_val));
    assign corr_a = delta_a * w_val;
    assign corr_b = a_val * delta_b;
endmodule

// Correction Unit: four 1-of-16 multiplexers + the correction arithmetic.
module correction_unit(
    input  [4*16-1:0] a_flat,
    input  [4*16-1:0] w_flat,
    input  [3:0] index_a, input [1:0] mant_a,
    input  [3:0] index_b, input [1:0] mant_b,
    output signed [12:0] corr_a,
    output signed [12:0] corr_b
);
    wire [3:0] a_at_ia, w_at_ia, a_at_ib, w_at_ib;
    sel16 M0(.flat(a_flat), .index(index_a), .code(a_at_ia));
    sel16 M1(.flat(w_flat), .index(index_a), .code(w_at_ia));
    sel16 M2(.flat(a_flat), .index(index_b), .code(a_at_ib));
    sel16 M3(.flat(w_flat), .index(index_b), .code(w_at_ib));
    correction_arith AR(.a_at_ia(a_at_ia), .w_at_ia(w_at_ia), .a_at_ib(a_at_ib), .w_at_ib(w_at_ib),
                        .mant_a(mant_a), .mant_b(mant_b), .corr_a(corr_a), .corr_b(corr_b));
endmodule

// 18-operand carry-save tree: 16 main products (level 1) + 2 correction products (join later).
module tree18(
    input  [16*21-1:0] prods,
    input  signed [12:0] corr_a,
    input  signed [12:0] corr_b,
    output signed [20:0] sum
);
    wire [20:0] x[0:15];
    genvar i;
    generate for (i = 0; i < 16; i = i + 1) begin : X assign x[i] = prods[i*21 +: 21]; end endgenerate
    wire [20:0] ca = {{8{corr_a[12]}}, corr_a};
    wire [20:0] cb = {{8{corr_b[12]}}, corr_b};
    wire [20:0] s0, c0;  csa #(21) u0(.a(x[0]), .b(x[1]), .c(x[2]), .s(s0), .cy(c0));
    wire [20:0] s1, c1;  csa #(21) u1(.a(x[3]), .b(x[4]), .c(x[5]), .s(s1), .cy(c1));
    wire [20:0] s2, c2;  csa #(21) u2(.a(x[6]), .b(x[7]), .c(x[8]), .s(s2), .cy(c2));
    wire [20:0] s3, c3;  csa #(21) u3(.a(x[9]), .b(x[10]), .c(x[11]), .s(s3), .cy(c3));
    wire [20:0] s4, c4;  csa #(21) u4(.a(x[12]), .b(x[13]), .c(x[14]), .s(s4), .cy(c4));
    wire [20:0] s5, c5;  csa #(21) u5(.a(s0), .b(c0), .c(s1), .s(s5), .cy(c5));
    wire [20:0] s6, c6;  csa #(21) u6(.a(c1), .b(s2), .c(c2), .s(s6), .cy(c6));
    wire [20:0] s7, c7;  csa #(21) u7(.a(s3), .b(c3), .c(s4), .s(s7), .cy(c7));
    wire [20:0] s8, c8;  csa #(21) u8(.a(c4), .b(x[15]), .c(ca), .s(s8), .cy(c8));
    wire [20:0] s9, c9;  csa #(21) u9(.a(s5), .b(c5), .c(s6), .s(s9), .cy(c9));
    wire [20:0] s10, c10;  csa #(21) u10(.a(c6), .b(s7), .c(c7), .s(s10), .cy(c10));
    wire [20:0] s11, c11;  csa #(21) u11(.a(s8), .b(c8), .c(cb), .s(s11), .cy(c11));
    wire [20:0] s12, c12;  csa #(21) u12(.a(s9), .b(c9), .c(s10), .s(s12), .cy(c12));
    wire [20:0] s13, c13;  csa #(21) u13(.a(c10), .b(s11), .c(c11), .s(s13), .cy(c13));
    wire [20:0] s14, c14;  csa #(21) u14(.a(s12), .b(c12), .c(s13), .s(s14), .cy(c14));
    wire [20:0] s15, c15;  csa #(21) u15(.a(s14), .b(c14), .c(c13), .s(s15), .cy(c15));
    assign sum = s15 + c15;   // single carry-propagate adder
endmodule

// Crescent DPU: the correction products are two more operands of the adder tree.
module dpe_corrected(
    input  [4*16-1:0] a_flat,
    input  [4*16-1:0] w_flat,
    input  [3:0] index_a, input [1:0] mant_a,
    input  [3:0] index_b, input [1:0] mant_b,
    input  signed [15:0] s_a,
    input  signed [15:0] s_b,
    input  signed [31:0] c_in,
    output signed [31:0] acc_out
);
    wire [16*21-1:0] prods;
    wire signed [12:0] corr_a, corr_b;
    wire signed [20:0] combined;
    products16 #(21) PR(.a_flat(a_flat), .w_flat(w_flat), .prods(prods));
    correction_unit CU(.a_flat(a_flat), .w_flat(w_flat),
                       .index_a(index_a), .mant_a(mant_a),
                       .index_b(index_b), .mant_b(mant_b),
                       .corr_a(corr_a), .corr_b(corr_b));
    tree18 T(.prods(prods), .corr_a(corr_a), .corr_b(corr_b), .sum(combined));
    scale_acc #(21) SA(.sum(combined), .s_a(s_a), .s_b(s_b), .c_in(c_in), .acc_out(acc_out));
endmodule
