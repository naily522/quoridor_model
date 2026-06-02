#include "quoridor.hpp"
#include "player.hpp"
#include "view.hpp"
#include <cstdlib>

int main(int argc, char* argv[])
{
    // 权重文件路径 (可通过命令行参数指定，默认 rl/weights/quoridor_v1.weights)
    const char* weights_path = argc > 1 ? argv[1] : "rl/weights/quoridor_v1.weights";

    // ── 创建玩家 ──
    HumanPlayer human;
    RLPlayer ai;

    std::cout << "加载 AI 权重: " << weights_path << " ...\n";
    if (!ai.load_weights(weights_path)) {
        std::cerr << "权重加载失败，请先运行训练导出权重文件。\n";
        std::cerr << "用法: " << argv[0] << " [权重文件路径]\n";
        return 1;
    }
    std::cout << "AI 权重加载成功!\n\n";

    // 选择先后手
    std::cout << "选择先后手:\n";
    std::cout << "[1] 你先手 (Player 1, 上方)\n";
    std::cout << "[2] AI 先手 (Player 1, 上方)\n";
    char side = _getch();
    std::cout << side << "\n\n";

    Player* players[2];
    if (side == '1' || side == '1') {  // 人 Player 1 (上), AI Player 2 (下)
        players[0] = &human;
        players[1] = &ai;
    } else {                            // AI Player 1 (上), 人 Player 2 (下)
        players[0] = &ai;
        players[1] = &human;
    }

    Quoridor::State state;
    state.reset();
    int step = 0;

    while (true)
    {
        system("cls");
        display_board(state);
        std::cout << "\n第 " << (step + 1) << " 步 — 玩家 " << state.turn;
        if (players[state.turn - 1] == &ai) std::cout << " (AI)";
        else std::cout << " (你)";
        std::cout << "\n";

        // AI 额外输出估值信息
        if (players[state.turn - 1] == &ai) {
            int top_idx;
            float top_p = ai.get_top_prob(top_idx);
            auto top_action = RLPlayer::decode_action(top_idx);
            std::cout << "  AI 估值: " << ai.get_value() << "\n";
            std::cout << "  AI 首选: " << (top_action.isWall ? "放墙" : "移动")
                      << " (" << top_action.pos.first << "," << top_action.pos.second << ")"
                      << " 概率=" << (top_p * 100) << "%\n";
        }

        Player* cur = players[state.turn - 1];

        if (cur->choose_type(state) == 'm')
        {
            while (!cur->choose_move(state).apply(state));
        }
        else
        {
            while (!cur->choose_wall(state).apply(state));
        }

        step++;

        // 终局判断
        if (state.pos[0].first == 2 * Quoridor::ROW_SIZE - 1)
        {
            system("cls");
            display_board(state);
            std::cout << "\n玩家 1 获胜!\n";
            if (players[0] == &ai) std::cout << "AI 赢了!\n";
            else std::cout << "你赢了!\n";
            break;
        }
        if (state.pos[1].first == 1)
        {
            system("cls");
            display_board(state);
            std::cout << "\n玩家 2 获胜!\n";
            if (players[1] == &ai) std::cout << "AI 赢了!\n";
            else std::cout << "你赢了!\n";
            break;
        }

        if (step >= 200) {
            std::cout << "平局!\n";
            break;
        }
    }

    std::cout << "\n按任意键退出...";
    _getch();
    return 0;
}
