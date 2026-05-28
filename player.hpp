#ifndef PLAYER_HPP
#define PLAYER_HPP

#include "quoridor.hpp"
#include <iostream>
#include <conio.h>
#include <cstdlib>
#include <vector>

class Player
{
public:
    virtual char choose_type(const Quoridor::State& state) = 0;
    virtual Quoridor::Action choose_move(const Quoridor::State& state) = 0;
    virtual Quoridor::Action choose_wall(const Quoridor::State& state) = 0;
    virtual ~Player();
};

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

class RLPlayer : public Player
{
public:
    RLPlayer();

    virtual void encode_state(const Quoridor::State& state);
    virtual void get_action_probs();

    static constexpr int ACTION_NUM = 164;

    char choose_type(const Quoridor::State& state) override;
    Quoridor::Action choose_move(const Quoridor::State& state) override;
    Quoridor::Action choose_wall(const Quoridor::State& state) override;

protected:
    std::vector<float> action_probs;

private:
    Quoridor::Action cached_action;

    void compute_action(const Quoridor::State& state);

    struct ActionSet {
        static Quoridor::Action decode(int id, const Quoridor::State& s);
    };
};

RLPlayer::RLPlayer() : cached_action({0,0}, false, 0) {}

void RLPlayer::compute_action(const Quoridor::State& state)
{
    // 1) 编码局面并获取 164 个动作的概率
    encode_state(state);
    action_probs.assign(ACTION_NUM, 0);
    get_action_probs();

    // 2) 归一化 → 概率总和 = 1
    float sum = 0;
    for (int i = 0; i < ACTION_NUM; i++) sum += action_probs[i];
    if (sum > 0)
        for (int i = 0; i < ACTION_NUM; i++) action_probs[i] /= sum;

    // 3) 按概率分布采样（概率越高越容易被选中）
    float r = (float)rand() / RAND_MAX;
    float cum = 0;
    int id = ACTION_NUM - 1;
    for (int i = 0; i < ACTION_NUM; i++) {
        cum += action_probs[i];
        if (r < cum) { id = i; break; }
    }

    // 4) 解码为具体的 Action
    cached_action = ActionSet::decode(id, state);
}

char RLPlayer::choose_type(const Quoridor::State& state)
{
    compute_action(state);
    return cached_action.isWall ? 'w' : 'm';
}

Quoridor::Action RLPlayer::choose_move(const Quoridor::State& state)
{
    // 移动可能被墙挡住，重新采样避免死循环
    compute_action(state);
    return cached_action;
}

Quoridor::Action RLPlayer::choose_wall(const Quoridor::State& state)
{
    // 放墙可能失败，每次重采样避免死循环
    compute_action(state);
    return cached_action;
}

#endif