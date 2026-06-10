#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "quoridor.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>

// =============================================================================
// 神经网络推理 — network.hpp
//
// 功能:
//   纯 C++ 实现的卷积神经网络推理，无外部依赖。
//   加载由 Python export_weights() 导出的 .weights 文件，
//   执行前向传播得到策略分布和价值估计。
//
// 架构 (与 model.py 严格一致):
//   conv_input → 5×ResBlock → policy_head + value_head
//
// BatchNorm 已在 Python 端融合到前一层 Conv2d 的权重中，
// C++ 端只做 Conv2d → ReLU → Conv2d → ... → FC → Softmax/Tanh。
// =============================================================================

// ─── 网络结构常量 (必须与 config.py / model.py 保持严格一致) ───
#ifndef NET_C
#define NET_C 64       // conv_channels (v3)
#define NET_PC 32      // policy_channels
#define NET_VH 128     // value_hidden (v3)
#define NET_RES 8      // res_blocks (v3)
#define NET_H 9
#define NET_W 9
#endif

// =============================================================================
// 基础算子 (inline 以减少函数调用开销)
// =============================================================================

// 3×3 Conv2d with padding=1, stride=1
// 输入: in[C][H][W]  权重: out[C][in][3][3]  偏置: out[F]
// 输出: out[F][H][W]
inline void conv3x3(const float* input, int C, int H, int W,
                     const float* weight, const float* bias, int F,
                     float* output) {
    for (int f = 0; f < F; f++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float sum = bias ? bias[f] : 0.0f;
                for (int c = 0; c < C; c++) {
                    for (int kh = -1; kh <= 1; kh++) {
                        int ih = h + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = -1; kw <= 1; kw++) {
                            int iw = w + kw;
                            if (iw < 0 || iw >= W) continue;
                            // weight layout: [F][C][3][3]
                            int widx = ((f * C + c) * 3 + (kh + 1)) * 3 + (kw + 1);
                            int iidx = (c * H + ih) * W + iw;
                            sum += weight[widx] * input[iidx];
                        }
                    }
                }
                output[(f * H + h) * W + w] = sum;
            }
        }
    }
}

// 1×1 Conv2d (value head 使用)
inline void conv1x1(const float* input, int C, int H, int W,
                     const float* weight, const float* bias, int F,
                     float* output) {
    for (int f = 0; f < F; f++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                float sum = bias ? bias[f] : 0.0f;
                for (int c = 0; c < C; c++) {
                    int widx = f * C + c; // [F][C][1][1]
                    int iidx = (c * H + h) * W + w;
                    sum += weight[widx] * input[iidx];
                }
                output[(f * H + h) * W + w] = sum;
            }
        }
    }
}

// 全连接层: out[M] = weight[M][N] · in[N] + bias[M]
inline void fc(const float* input, int N,
                const float* weight, const float* bias, int M,
                float* output) {
    for (int m = 0; m < M; m++) {
        float sum = bias ? bias[m] : 0.0f;
        for (int n = 0; n < N; n++) {
            sum += weight[m * N + n] * input[n];
        }
        output[m] = sum;
    }
}

inline void relu(float* data, int n) {
    for (int i = 0; i < n; i++)
        if (data[i] < 0) data[i] = 0;
}

inline void tanh_(float* data, int n) {
    for (int i = 0; i < n; i++)
        data[i] = std::tanh(data[i]);
}

inline void softmax(float* data, int n) {
    float max_val = *std::max_element(data, data + n);
    float sum = 0;
    for (int i = 0; i < n; i++) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }
    if (sum > 1e-12f)
        for (int i = 0; i < n; i++) data[i] /= sum;
    else {
        float uniform = 1.0f / n;
        for (int i = 0; i < n; i++) data[i] = uniform;
    }
}

// =============================================================================
// 残差块: out = ReLU(conv2(ReLU(conv1(x))) + x)
// =============================================================================
inline void res_block(const float* input, int C, int H, int W,
                       const float* w1, const float* b1,
                       const float* w2, const float* b2,
                       float* buf, float* output) {
    // buf = conv1(x) → ReLU
    conv3x3(input, C, H, W, w1, b1, C, buf);
    relu(buf, C * H * W);

    // output = conv2(buf) + x → ReLU
    conv3x3(buf, C, H, W, w2, b2, C, output);
    for (int i = 0; i < C * H * W; i++)
        output[i] += input[i];
    relu(output, C * H * W);
}

// =============================================================================
// NetworkWeights — 加载和存储所有网络参数
// =============================================================================
struct NetworkWeights {
    // ── conv_input (7 input channels: absolute encoding) ──
    std::vector<float> conv_input_w;   // [32, 7, 3, 3]
    std::vector<float> conv_input_b;   // [32]

    // ── N× ResBlock (NET_RES = 8, 与 config.py 一致) ──
    struct RB {
        std::vector<float> w1, b1;     // [NET_C, NET_C, 3, 3], [NET_C]
        std::vector<float> w2, b2;     // [NET_C, NET_C, 3, 3], [NET_C]
    };
    RB res[NET_RES];

    // ── policy head ──
    std::vector<float> policy_conv_w;  // [32,32,3,3]
    std::vector<float> policy_conv_b;  // [32]
    std::vector<float> policy_fc_w;    // [225, 2592]
    std::vector<float> policy_fc_b;    // [225]

    // ── value head ──
    std::vector<float> value_conv_w;   // [1,32,1,1]
    std::vector<float> value_conv_b;   // [1]
    std::vector<float> value_fc0_w;    // [64, 81]
    std::vector<float> value_fc0_b;    // [64]
    std::vector<float> value_fc2_w;    // [1, 64]
    std::vector<float> value_fc2_b;    // [1]

    // 读取 n 个 float 到向量
    static void read_floats(std::ifstream& f, std::vector<float>& v, size_t n) {
        v.resize(n);
        f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    }

    bool load(const char* path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { std::cerr << "[Network] Cannot open: " << path << "\n"; return false; }

        // conv_input: [NET_C, 7, 3, 3] + [NET_C]
        read_floats(f, conv_input_w, NET_C * 7 * 3 * 3);
        read_floats(f, conv_input_b, NET_C);

        // NET_RES× ResBlock: each [NET_C, NET_C, 3, 3] + [NET_C] × 2
        for (int i = 0; i < NET_RES; i++) {
            read_floats(f, res[i].w1, NET_C * NET_C * 3 * 3);
            read_floats(f, res[i].b1, NET_C);
            read_floats(f, res[i].w2, NET_C * NET_C * 3 * 3);
            read_floats(f, res[i].b2, NET_C);
        }

        // policy_conv: [NET_PC, NET_C, 3, 3] + [NET_PC]
        read_floats(f, policy_conv_w, NET_PC * NET_C * 3 * 3);
        read_floats(f, policy_conv_b, NET_PC);
        // policy_fc: [225, NET_PC * 9 * 9] + [225]
        read_floats(f, policy_fc_w, 225 * NET_PC * NET_H * NET_W);
        read_floats(f, policy_fc_b, 225);

        // value_conv: [1, NET_C, 1, 1] + [1]
        read_floats(f, value_conv_w, 1 * NET_C * 1 * 1);
        read_floats(f, value_conv_b, 1);
        // value_fc0: [NET_VH, NET_H * NET_W] + [NET_VH]
        read_floats(f, value_fc0_w, NET_VH * NET_H * NET_W);
        read_floats(f, value_fc0_b, NET_VH);
        // value_fc2: [1, NET_VH] + [1]
        read_floats(f, value_fc2_w, 1 * NET_VH);
        read_floats(f, value_fc2_b, 1);

        bool ok = f.good();
        f.close();
        if (!ok) { std::cerr << "[Network] Incomplete weight file\n"; return false; }
        return true;
    }
};

// =============================================================================
// 前向传播 — 输入编码后的 6×9×9 张量，输出 policy[225] 和 value
// =============================================================================
inline void forward(const NetworkWeights& w,
                     const float state[7][9][9],
                     float policy[225], float& value) {
    const int H = NET_H, W = NET_W, C = NET_C;

    // 工作缓冲区 (all allocated on stack for small size)
    float buf0[C * H * W];  // conv_input out
    float buf1[C * H * W];  // res_block temp
    float buf2[C * H * W];  // res_block out / final features

    // ── conv_input: [7,9,9] → [NET_C,9,9] ──
    conv3x3(&state[0][0][0], 7, H, W,
            w.conv_input_w.data(), w.conv_input_b.data(), C,
            buf0);
    relu(buf0, C * H * W);

    // ── NET_RES× ResBlock ──
    const float* res_in = buf0;
    float* res_buf = buf1;
    float* res_out = buf2;

    for (int i = 0; i < NET_RES; i++) {
        res_block(res_in, C, H, W,
                  w.res[i].w1.data(), w.res[i].b1.data(),
                  w.res[i].w2.data(), w.res[i].b2.data(),
                  res_buf, res_out);
        // 下一轮: 输出作为输入
        if (i < NET_RES - 1) {
            const float* tmp = res_in;
            res_in = res_out;
            res_out = const_cast<float*>(tmp);
            float* tmp_buf = res_buf;
            res_buf = res_out;
            res_out = tmp_buf;
        }
    }

    // ── policy head ──
    float p_conv_out[NET_PC * H * W];
    conv3x3(res_out, C, H, W,
            w.policy_conv_w.data(), w.policy_conv_b.data(), NET_PC,
            p_conv_out);
    relu(p_conv_out, NET_PC * H * W);

    // policy_fc: [NET_PC * H * W] → [225]
    fc(p_conv_out, NET_PC * H * W,
       w.policy_fc_w.data(), w.policy_fc_b.data(), 225,
       policy);
    softmax(policy, 225);

    // ── value head ──
    float v_conv_out[1 * H * W];
    conv1x1(res_out, C, H, W,
            w.value_conv_w.data(), w.value_conv_b.data(), 1,
            v_conv_out);
    relu(v_conv_out, 1 * H * W);

    // value_fc: [81] → [NET_VH] → [1]
    float v_hidden[NET_VH];
    fc(v_conv_out, H * W,
       w.value_fc0_w.data(), w.value_fc0_b.data(), NET_VH,
       v_hidden);
    relu(v_hidden, NET_VH);

    float v_raw[1];
    fc(v_hidden, NET_VH,
       w.value_fc2_w.data(), w.value_fc2_b.data(), 1,
       v_raw);
    tanh_(v_raw, 1);
    value = v_raw[0];
}

#endif // NETWORK_HPP
