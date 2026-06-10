#include "quoridor.hpp"
#include "player.hpp"
#include "view.hpp"
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iomanip>

// 1-ply 前瞻搜索: 对每个合法动作, 模拟执行后用价值网络评估后继局面,
// 选择对己方最有利的动作
Quoridor::Action ai_best_action(RLPlayer& ai, const Quoridor::State& state)
{
    auto legal = RLPlayer::get_all_legal_actions(state);

    // 先获取当前局面的原始策略分布 (作为先验)
    ai.encode_state(state);
    ai.get_action_probs(state);   // 填充 ai.get_probs()
    const auto& policy = ai.get_probs();

    float best_score = -1e9f;
    int best_idx = 0;

    // 展示 AI 的前几个候选
    std::vector<std::pair<float, int>> candidates;
    for (auto& a : legal) {
        int idx = RLPlayer::action_to_index(a);
        float prior = policy[idx];

        // 模拟走一步
        Quoridor::State next = state;
        if (!a.apply(next)) continue;

        // 价值网络评估后继局面 (next.turn 已切换为对手)
        float opponent_val = ai.evaluate_state(next);
        // 转为当前玩家视角: 对手好 = 我差
        float my_val = -opponent_val;

        // 综合评分: 价值主导 + 策略先验微调
        float score = my_val + 0.1f * std::log(prior + 1e-8f);

        candidates.push_back({score, idx});

        if (score > best_score) {
            best_score = score;
            best_idx = idx;
        }
    }

    // 按评分排序展示
    std::sort(candidates.begin(), candidates.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    std::cout << "\n  AI 搜索 (1-ply, " << legal.size() << " 个合法动作):\n";
    int show_n = std::min(5, (int)candidates.size());
    for (int i = 0; i < show_n; i++) {
        auto act = RLPlayer::decode_action(candidates[i].second);
        std::cout << "    [" << (i + 1) << "] "
                  << (act.isWall ? (act.wall_dir == 0 ? "垂直墙" : "水平墙") : "移动")
                  << " (" << act.pos.first << "," << act.pos.second << ")"
                  << "  score=" << std::fixed << std::setprecision(3)
                  << candidates[i].first << "\n";
    }

    return RLPlayer::decode_action(best_idx);
}


int main(int argc, char* argv[])
{
    // 修复: 初始化随机种子, 确保每次对局 AI 行为不同
    srand((unsigned)time(NULL));

    const char* weights_path = argc > 1 ? argv[1] : "rl/weights/quoridor_v3.weights";

    HumanPlayer human;
    RLPlayer ai;

    std::cout << "========================================\n";
    std::cout << "  步步为营 (Quoridor) — 人机对战\n";
    std::cout << "  AI 模型: v3 (64ch/8res/1.2M)  |  搜索: 1-ply value lookahead\n";
    std::cout << "========================================\n\n";

    std::cout << "Loading AI weights: " << weights_path << " ...\n";
    if (!ai.load_weights(weights_path)) {
        std::cerr << "Failed to load weights.\n";
        std::cerr << "Usage: " << argv[0] << " [weights_file_path]\n";
        return 1;
    }
    std::cout << "AI weights loaded successfully!\n\n";

    std::cout << "Choose your side:\n";
    std::cout << "  [1] You go first  (Player 1, top    → goes DOWN)\n";
    std::cout << "  [2] AI goes first (Player 1, top    → goes DOWN)\n";
    char side = _getch();
    std::cout << side << "\n\n";

    Player* players[2];
    if (side == '1') {
        players[0] = &human;   // 你先手 (Player 1)
        players[1] = &ai;      // AI 后手 (Player 2)
    } else {
        players[0] = &ai;      // AI 先手 (Player 1)
        players[1] = &human;   // 你后手 (Player 2)
    }

    Quoridor::State state;
    state.reset();
    int step = 0;

    while (true)
    {
        system("cls");
        display_board(state);

        std::cout << "\n  Step " << (step + 1) << "  |  Player " << state.turn;
        if (players[state.turn - 1] == &ai) std::cout << " (AI)";
        else std::cout << " (You)";
        std::cout << "  |  Walls: P1=" << state.wall_num[0]
                  << " P2=" << state.wall_num[1] << "\n";

        Player* cur = players[state.turn - 1];

        if (cur == &ai) {
            // AI: 1-ply 搜索决策
            Quoridor::Action ai_action = ai_best_action(ai, state);
            std::cout << "\n  AI chooses: "
                      << (ai_action.isWall
                          ? (ai_action.wall_dir == 0 ? "垂直墙" : "水平墙")
                          : "移动")
                      << " (" << ai_action.pos.first << "," << ai_action.pos.second << ")\n";

            if (!ai_action.apply(state)) {
                std::cerr << "  ERROR: AI action failed! Trying fallback...\n";
                // 回退: 取第一个合法动作
                auto legal = RLPlayer::get_all_legal_actions(state);
                if (!legal.empty()) legal[0].apply(state);
            }
        }
        else {
            // 人类玩家
            if (cur->choose_type(state) == 'm')
                cur->choose_move(state).apply(state);
            else
                cur->choose_wall(state).apply(state);
        }

        step++;

        // 胜负判定
        if (state.pos[0].first == 2 * Quoridor::ROW_SIZE - 1)
        {
            system("cls");
            display_board(state);
            std::cout << "\n  *** Player 1 ";
            if (players[0] == &ai) std::cout << "(AI) ";
            else std::cout << "(You) ";
            std::cout << "wins! ***\n";
            break;
        }
        if (state.pos[1].first == 1)
        {
            system("cls");
            display_board(state);
            std::cout << "\n  *** Player 2 ";
            if (players[1] == &ai) std::cout << "(AI) ";
            else std::cout << "(You) ";
            std::cout << "wins! ***\n";
            break;
        }

        if (step >= 200) {
            system("cls");
            display_board(state);
            std::cout << "\n  *** Draw (max steps)! ***\n";
            break;
        }
    }

    std::cout << "\n  Total steps: " << step;
    std::cout << "\n\n  Press any key to exit...";
    _getch();
    return 0;
}
