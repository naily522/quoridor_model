#ifndef PLAYER_HPP
#define PLAYER_HPP

#include "quoridor.hpp"
#include "network.hpp"
#include <iostream>
#include <conio.h>
#include <cstdlib>
#include <vector>
#include <cstring>

class Player
{
public:
    virtual char choose_type(const Quoridor::State& state) = 0;
    virtual Quoridor::Action choose_move(const Quoridor::State& state) = 0;
    virtual Quoridor::Action choose_wall(const Quoridor::State& state) = 0;
    virtual ~Player();
};

// =============================================================================
// HumanPlayer — 键盘输入
// =============================================================================
class HumanPlayer : public Player
{
public:
    char choose_type(const Quoridor::State& state) override
    {
        auto& cur = state.pos[state.turn-1];
        std::cout << "Player " << state.turn << " at (" << cur.first << "," << cur.second << ")\n";
        if (state.wall_num[state.turn-1] > 0) std::cout << "[M]ove  [W]all: ";
        else std::cout << "[M]ove (no walls left): ";

        char c = _getch();
        std::cout << c << "\n";
        if (c == 'm' || c == 'M') return 'm';
        if ((c == 'w' || c == 'W') && state.wall_num[state.turn-1] > 0) return 'w';
        return choose_type(state);
    }

    Quoridor::Action choose_move(const Quoridor::State& state) override
    {
        auto& cur = state.pos[state.turn-1];
        auto& opp = state.pos[2-state.turn];
        int dirs[4][2] = {{-2,0},{2,0},{0,-2},{0,2}};
        vector<pair<int,int>> targets;
        char key = 'a';

        for (auto& d : dirs) {
            int nr = cur.first + d[0], nc = cur.second + d[1];
            if (state.board[cur.first + d[0]/2][cur.second + d[1]/2] == 1) continue;
            if (nr < 0 || nr > 2*Quoridor::ROW_SIZE || nc < 0 || nc > 2*Quoridor::COLUMN_SIZE) continue;
            if (nr%2==0 || nc%2==0) continue;

            if (nr == opp.first && nc == opp.second) {
                for (auto& dj : dirs) {
                    int jr = opp.first + dj[0], jc = opp.second + dj[1];
                    if (jr == cur.first && jc == cur.second) continue;
                    if (jr < 0 || jr > 2*Quoridor::ROW_SIZE || jc < 0 || jc > 2*Quoridor::COLUMN_SIZE) continue;
                    if (jr%2==0 || jc%2==0) continue;
                    if (state.board[opp.first + dj[0]/2][opp.second + dj[1]/2] == 1) continue;
                    std::cout << "[" << key++ << "] (" << jr << "," << jc << ")\n";
                    targets.push_back({jr, jc});
                }
            } else {
                std::cout << "[" << key++ << "] (" << nr << "," << nc << ")\n";
                targets.push_back({nr, nc});
            }
        }

        while (true) {
            char c = _getch();
            int idx = c - 'a';
            if (idx >= 0 && idx < (int)targets.size())
                return {targets[idx], false, 0};
        }
    }

    Quoridor::Action choose_wall(const Quoridor::State& state) override
    {
        int r, c, d;
        std::cout << "Row (0-" << 2*Quoridor::ROW_SIZE << "): "; std::cin >> r;
        std::cout << "Col (0-" << 2*Quoridor::COLUMN_SIZE << "): "; std::cin >> c;
        std::cout << "Dir (0=vert,1=horiz): "; std::cin >> d;
        return {{r, c}, true, d};
    }
};

Player::~Player() {}

// =============================================================================
// RLPlayer — 神经网络 AI
// =============================================================================
class RLPlayer : public Player
{
public:
    RLPlayer() : cached_action({0,0}, false, 0) {}

    bool load_weights(const char* path) {
        return weights.load(path);
    }

    // ─── 局面编码 (6 通道 9×9, 与 encode.py 一致) ───
    void encode_state(const Quoridor::State& state) {
        std::memset(encoded, 0, sizeof(encoded));

        // ch 0: current player position
        auto my = state.pos[state.turn - 1];
        encoded[0][my.first / 2][my.second / 2] = 1.0f;

        // ch 1: opponent position
        auto opp = state.pos[2 - state.turn];
        encoded[1][opp.first / 2][opp.second / 2] = 1.0f;

        // ch 2: current player's remaining walls
        float my_walls = state.wall_num[state.turn - 1] / 10.0f;
        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
                encoded[2][i][j] = my_walls;

        // ch 3: opponent's remaining walls
        float opp_walls = state.wall_num[2 - state.turn] / 10.0f;
        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
                encoded[3][i][j] = opp_walls;

        // ch 4: horizontal walls
        for (int wr = 0; wr < 9; wr++)
            for (int wc = 0; wc < 9; wc++)
                if (state.h_wall[wr][wc])
                    encoded[4][wr][wc] = 1.0f;

        // ch 5: vertical walls
        for (int wr = 0; wr < 9; wr++)
            for (int wc = 0; wc < 9; wc++)
                if (state.v_wall[wr][wc])
                    encoded[5][wr][wc] = 1.0f;
    }

    // ─── 动作索引解码 (0~224 → Action, 与 index_to_action 一致) ───
    static Quoridor::Action decode_action(int idx) {
        if (idx < 81) {  // 移动
            int r = (idx / 9) * 2 + 1;
            int c = (idx % 9) * 2 + 1;
            return {{r, c}, false, 0};
        } else if (idx < 153) {  // 垂直墙
            int wall_idx = idx - 81;
            int r = (wall_idx / 9) * 2 + 2;
            int c = (wall_idx % 9) * 2 + 2;
            return {{r, c}, true, 0};
        } else {  // 水平墙
            int wall_idx = idx - 153;
            int r = (wall_idx / 8) * 2 + 2;
            int c = (wall_idx % 8) * 2 + 2;
            return {{r, c}, true, 1};
        }
    }

    // ─── 获取动作概率 (网络推理 + 合法动作掩码) ───
    void get_action_probs(const Quoridor::State& state) {
        // 前向传播
        forward(weights, encoded, raw_policy, value);

        // 合法性掩码 + 重归一化
        action_probs.assign(ACTION_NUM, 0.0f);

        int legal_count = 0;
        for (auto& a : get_all_legal_actions(state)) {
            int idx = action_to_index(a);
            action_probs[idx] = raw_policy[idx];
            legal_count++;
        }

        float sum = 0;
        for (float p : action_probs) sum += p;
        if (sum > 1e-12f) {
            for (float& p : action_probs) p /= sum;
        } else {
            // fallback: 均匀分布
            float uniform = 1.0f / legal_count;
            for (float& p : action_probs)
                if (p > 0) p = uniform;
        }
    }

    // ─── 工具: Action → 索引 ───
    static int action_to_index(const Quoridor::Action& a) {
        int r_idx = (a.pos.first - 1) / 2;
        int c_idx = (a.pos.second - 1) / 2;
        if (a.isWall) {
            if (a.wall_dir == 0)      // 垂直墙
                return 81 + r_idx * 9 + c_idx;
            else                      // 水平墙
                return 153 + r_idx * 8 + c_idx;
        } else {
            return r_idx * 9 + c_idx;
        }
    }

    // ─── 获取合法动作列表 (与 get_legal_actions 相同) ───
    static vector<Quoridor::Action> get_all_legal_actions(const Quoridor::State& state) {
        vector<Quoridor::Action> result;

        auto& cur = state.pos[state.turn - 1];
        auto& opp = state.pos[2 - state.turn];
        int dirs[4][2] = {{-2,0},{2,0},{0,-2},{0,2}};

        // 移动
        for (auto& d : dirs) {
            int nr = cur.first + d[0], nc = cur.second + d[1];
            int wall_r = cur.first + d[0]/2, wall_c = cur.second + d[1]/2;
            if (nr < 0 || nr > 2*Quoridor::ROW_SIZE || nc < 0 || nc > 2*Quoridor::COLUMN_SIZE) continue;
            if (nr % 2 == 0 || nc % 2 == 0) continue;
            if (state.board[wall_r][wall_c] == 1) continue;

            if (nr == opp.first && nc == opp.second) {
                // 跳过对方
                for (auto& dj : dirs) {
                    int jr = opp.first + dj[0], jc = opp.second + dj[1];
                    int jwall_r = opp.first + dj[0]/2, jwall_c = opp.second + dj[1]/2;
                    if (jr == cur.first && jc == cur.second) continue;
                    if (jr < 0 || jr > 2*Quoridor::ROW_SIZE || jc < 0 || jc > 2*Quoridor::COLUMN_SIZE) continue;
                    if (jr % 2 == 0 || jc % 2 == 0) continue;
                    if (state.board[jwall_r][jwall_c] == 1) continue;
                    result.push_back({{jr, jc}, false, 0});
                }
            } else {
                result.push_back({{nr, nc}, false, 0});
            }
        }

        // 放墙
        if (state.wall_num[state.turn - 1] > 0) {
            for (int r = 0; r <= 2*Quoridor::ROW_SIZE; r += 2) {
                for (int c = 0; c <= 2*Quoridor::COLUMN_SIZE; c += 2) {
                    if (r - (Quoridor::WALL_LENGTH-1) >= 0 &&
                        r + (Quoridor::WALL_LENGTH-1) <= 2*Quoridor::ROW_SIZE) {
                        if (state.board[r][c] == 0 && state.board[r+1][c] == 0) {
                            Quoridor::State tmp = state;
                            bool ok = true;
                            for (int i = -(Quoridor::WALL_LENGTH-1); i < Quoridor::WALL_LENGTH; i++) {
                                if (tmp.board[r + i][c]) { ok = false; break; }
                                tmp.board[r + i][c] = 1;
                            }
                            if (ok && isconnect(tmp.board, state.pos[0], 2*Quoridor::ROW_SIZE-1)
                                 && isconnect(tmp.board, state.pos[1], 1))
                                result.push_back({{r, c}, true, 0});
                        }
                    }
                    if (c - (Quoridor::WALL_LENGTH-1) >= 0 &&
                        c + (Quoridor::WALL_LENGTH-1) <= 2*Quoridor::COLUMN_SIZE) {
                        if (state.board[r][c] == 0 && state.board[r][c+1] == 0) {
                            Quoridor::State tmp = state;
                            bool ok = true;
                            for (int i = -(Quoridor::WALL_LENGTH-1); i < Quoridor::WALL_LENGTH; i++) {
                                if (tmp.board[r][c + i]) { ok = false; break; }
                                tmp.board[r][c + i] = 1;
                            }
                            if (ok && isconnect(tmp.board, state.pos[0], 2*Quoridor::ROW_SIZE-1)
                                 && isconnect(tmp.board, state.pos[1], 1))
                                result.push_back({{r, c}, true, 1});
                        }
                    }
                }
            }
        }

        return result;
    }

    // ─── Player 接口 ───
    char choose_type(const Quoridor::State& state) override {
        compute_action(state);
        return cached_action.isWall ? 'w' : 'm';
    }

    Quoridor::Action choose_move(const Quoridor::State& state) override {
        compute_action(state);
        return cached_action;
    }

    Quoridor::Action choose_wall(const Quoridor::State& state) override {
        compute_action(state);
        return cached_action;
    }

    // 获取调试信息
    float get_value() const { return value; }
    const std::vector<float>& get_probs() const { return action_probs; }
    float get_top_prob(int& out_idx) const {
        out_idx = 0;
        float max_p = action_probs[0];
        for (int i = 1; i < ACTION_NUM; i++) {
            if (action_probs[i] > max_p) {
                max_p = action_probs[i];
                out_idx = i;
            }
        }
        return max_p;
    }

    static constexpr int ACTION_NUM = Quoridor::ACTION_NUM;

private:
    NetworkWeights weights;
    float encoded[6][9][9];        // 编码后的输入张量
    float raw_policy[ACTION_NUM];  // 网络原始策略输出
    float value;                   // 网络价值输出
    std::vector<float> action_probs;  // 掩码后的合法动作概率
    Quoridor::Action cached_action;

    void compute_action(const Quoridor::State& state) {
        encode_state(state);
        get_action_probs(state);

        // 按概率采样
        float r = (float)rand() / RAND_MAX;
        float cum = 0;
        int id = ACTION_NUM - 1;
        for (int i = 0; i < ACTION_NUM; i++) {
            cum += action_probs[i];
            if (r < cum) { id = i; break; }
        }
        cached_action = decode_action(id);
    }
};

#endif // PLAYER_HPP
