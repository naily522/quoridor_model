#include "quoridor.hpp"
#include "player.hpp"
#include "view.hpp"
#include <cstdlib>

int main(int argc, char* argv[])
{
    const char* weights_path = argc > 1 ? argv[1] : "rl/weights/quoridor_v1.weights";

    HumanPlayer human;
    RLPlayer ai;

    std::cout << "Loading AI weights: " << weights_path << " ...\n";
    if (!ai.load_weights(weights_path)) {
        std::cerr << "Failed to load weights. Run training first to export weights.\n";
        std::cerr << "Usage: " << argv[0] << " [weights_file_path]\n";
        return 1;
    }
    std::cout << "AI weights loaded successfully!\n\n";

    std::cout << "Choose your side:\n";
    std::cout << "[1] You go first (Player 1, top)\n";
    std::cout << "[2] AI goes first (Player 1, top)\n";
    char side = _getch();
    std::cout << side << "\n\n";

    Player* players[2];
    if (side == '1') {
        players[0] = &human;
        players[1] = &ai;
    } else {
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
        std::cout << "\nStep " << (step + 1) << " - Player " << state.turn;
        if (players[state.turn - 1] == &ai) std::cout << " (AI)";
        else std::cout << " (You)";
        std::cout << "\n";

        if (players[state.turn - 1] == &ai) {
            int top_idx;
            float top_p = ai.get_top_prob(top_idx);
            auto top_action = RLPlayer::decode_action(top_idx);
            std::cout << "  AI eval: " << ai.get_value() << "\n";
            std::cout << "  AI best: " << (top_action.isWall ? "wall" : "move")
                      << " (" << top_action.pos.first << "," << top_action.pos.second << ")"
                      << " prob=" << (top_p * 100) << "%\n";
        }

        Player* cur = players[state.turn - 1];

        if (cur->choose_type(state) == 'm')
            while (!cur->choose_move(state).apply(state));
        else
            while (!cur->choose_wall(state).apply(state));

        step++;

        if (state.pos[0].first == 2 * Quoridor::ROW_SIZE - 1)
        {
            system("cls");
            display_board(state);
            std::cout << "\nPlayer 1 wins!\n";
            if (players[0] == &ai) std::cout << "AI wins!\n";
            else std::cout << "You win!\n";
            break;
        }
        if (state.pos[1].first == 1)
        {
            system("cls");
            display_board(state);
            std::cout << "\nPlayer 2 wins!\n";
            if (players[1] == &ai) std::cout << "AI wins!\n";
            else std::cout << "You win!\n";
            break;
        }

        if (step >= 200) {
            std::cout << "Draw!\n";
            break;
        }
    }

    std::cout << "\nPress any key to exit...";
    _getch();
    return 0;
}
