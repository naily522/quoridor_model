#ifndef QUORIDOR_HPP
#define QUORIDOR_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <queue>

using std::pair;
using std::vector;
using std::string;
using std::queue;

/******************* Environment Definition *******************/
class Quoridor
{
public:
    static constexpr int ROW_SIZE=9, COLUMN_SIZE=9, WALL_LENGTH=2, WALL_NUM=10;

    bool verbose;

    struct State
    {
        pair<int, int> pos[2];
        int board[2*ROW_SIZE+1][2*COLUMN_SIZE+1]; // (odd,odd)=square; (even,*)=wall/channel
        int turn, wall_num[2]; // turn=1,2 for player1, player2; wall_num for remaining walls

        void reset();
    };

    struct Action
    {
        pair<int, int> pos; // target pos for move, or wall start coordinate for wall
        bool isWall;
        int wall_dir; // 0: vertical, 1: horizoncal

        Action(pair<int, int> p, bool w, int d) : pos(p), isWall(w), wall_dir(d) {}
        bool apply(State& state) const;
    };
    
    void log_msg(const string& msg)
    { if (verbose) log.push_back(msg); }

    void flush_log(const string& path)
    {
        std::ofstream f(path);
        for(auto& l : log) f << l << std::endl;
    }

    void clear_log()
    { log.clear(); }
private:
    vector<string> log;
};

void Quoridor::State::reset()
{
    memset(this->board, 0, sizeof(this->board));
    // boundary walls (outer ring)
    for (int i = 0; i < 2*ROW_SIZE+1; i++)
    {
        this->board[i][0] = 1;
        this->board[i][2*COLUMN_SIZE] = 1;
    }
    for (int j = 0; j < 2*COLUMN_SIZE+1; j++)
    {
        this->board[0][j] = 1;
        this->board[2*ROW_SIZE][j] = 1;
    }
    this->turn = 1;
    this->wall_num[0] = this->wall_num[1] = WALL_NUM;
    this->pos[0] = {1, COLUMN_SIZE + (1-COLUMN_SIZE%2)};         // top center
    this->pos[1] = {2*ROW_SIZE-1, COLUMN_SIZE + (1-COLUMN_SIZE%2)}; // bottom center
}

bool Quoridor::Action::apply(State& state) const
{
    if (this->isWall)
    {
        if (this->pos.first % 2 != 0 || this->pos.second % 2 != 0) return false;

        State temp_state = state;
        // do walls overlap with existing wall?
        if (this->wall_dir == 0) // vertical
        {
            for (int i = -(WALL_LENGTH-1); i < WALL_LENGTH; i++)
            {
                if (temp_state.board[this->pos.first + i][this->pos.second] == 1) return false;
                temp_state.board[this->pos.first + i][this->pos.second] = 1;
            }
        }
        else // horizontal
        {
            for (int i = -(WALL_LENGTH-1); i < WALL_LENGTH; i++)
            {
                if (temp_state.board[this->pos.first][this->pos.second + i] == 1) return false;
                temp_state.board[this->pos.first][this->pos.second + i] = 1;
            }
        }

        // check if the wall blocks all paths for either player
        bool isconnect(const int board[2*ROW_SIZE+1][2*COLUMN_SIZE+1], pair<int, int> start, int target_row);

        if (!isconnect(temp_state.board, state.pos[0], 2*ROW_SIZE-1) || !isconnect(temp_state.board, state.pos[1], 1)) return false;

        state = temp_state; // update state with the new wall
        state.wall_num[state.turn-1]--; // decrease wall count for the player
    }
    else // move
    {
        state.pos[state.turn-1] = this->pos;
    }

    state.turn = 3 - state.turn;
    return true;
}

bool isconnect(const int board[2*Quoridor::ROW_SIZE+1][2*Quoridor::COLUMN_SIZE+1], pair<int, int> start, int target_row)
{
    queue<pair<int, int>> q;
    bool isvisited[2*Quoridor::ROW_SIZE+1][2*Quoridor::COLUMN_SIZE+1] = {0};
    q.push(start);
    while (!q.empty())
    {
        auto [x, y] = q.front();
        q.pop();
        if (x == target_row) return true;
        for (auto& [dx, dy] : vector<pair<int, int>>{{-2, 0}, {2, 0}, {0, -2}, {0, 2}})
        {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx > 2*Quoridor::ROW_SIZE || ny < 0 || ny > 2*Quoridor::COLUMN_SIZE) continue;
            if (board[x+(dx/2)][y+(dy/2)] == 1) continue; // wall
            if (isvisited[nx][ny]) continue;
            isvisited[nx][ny] = true;
            q.push({nx, ny});
        }
    }
    return false;
}

#endif