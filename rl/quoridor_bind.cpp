// =============================================================================
// pybind11 绑定 — quoridor_bind.cpp
//
// 功能:
//   将 quoridor.hpp 中的 C++ 类/函数暴露给 Python，
//   让 self_play.py、encode.py 等可以直接调用 C++ 游戏逻辑。
//
// 编译后用法:
//   from quoridor_cpp import State, Action, isconnect, ...
//   s = State()
//   s.reset()
//   a = Action((3, 5), False, 0)
//   a.apply(s)
//
// 编译方法见 build_pyd.bat
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>        // std::pair, std::vector 自动转换
#include <pybind11/functional.h>
#include "../quoridor.hpp"

namespace py = pybind11;

// ─── 辅助函数：获取某一玩家的所有合法移动目标位置 ───
// 返回值: vector of (row, col)，每个坐标代表一步合法移动的落子位置
static std::vector<std::pair<int, int>> get_legal_moves_for_player(
    const Quoridor::State& state, int player)
{
    std::vector<std::pair<int, int>> result;
    auto& cur = state.pos[player - 1];
    auto& opp = state.pos[2 - player]; // opponent

    int dirs[4][2] = {{-2,0}, {2,0}, {0,-2}, {0,2}};
    for (auto& d : dirs) {
        int nr = cur.first + d[0], nc = cur.second + d[1];
        int wall_r = cur.first + d[0]/2, wall_c = cur.second + d[1]/2;

        // 出界
        if (nr < 0 || nr > 2*Quoridor::ROW_SIZE ||
            nc < 0 || nc > 2*Quoridor::COLUMN_SIZE)
            continue;
        // 不是可落子的格子 (奇数行奇数列才是格子)
        if (nr % 2 == 0 || nc % 2 == 0) continue;
        // 有墙挡住
        if (state.board[wall_r][wall_c] == 1) continue;

        // 对面是对方棋子 → 尝试跳过去
        if (nr == opp.first && nc == opp.second) {
            for (auto& dj : dirs) {
                int jr = opp.first + dj[0], jc = opp.second + dj[1];
                int jwall_r = opp.first + dj[0]/2, jwall_c = opp.second + dj[1]/2;

                if (jr == cur.first && jc == cur.second) continue; // 跳回起点
                if (jr < 0 || jr > 2*Quoridor::ROW_SIZE ||
                    jc < 0 || jc > 2*Quoridor::COLUMN_SIZE)
                    continue;
                if (jr % 2 == 0 || jc % 2 == 0) continue;
                if (state.board[jwall_r][jwall_c] == 1) continue;

                result.push_back({jr, jc});
            }
        } else {
            result.push_back({nr, nc});
        }
    }
    return result;
}

// ─── 辅助函数：获取所有合法放墙位置 ───
// 返回值: vector of (row, col, dir)，dir=0 垂直, dir=1 水平
static std::vector<std::tuple<int, int, int>> get_legal_walls_for_player(
    const Quoridor::State& state, int player)
{
    std::vector<std::tuple<int, int, int>> result;
    if (state.wall_num[player - 1] <= 0) return result;

    // 墙只能放在偶数坐标上 (墙通道的交叉点)
    for (int r = 1; r < 2*Quoridor::ROW_SIZE; r += 2) {
        for (int c = 1; c < 2*Quoridor::COLUMN_SIZE; c += 2) {
            // 尝试垂直墙 (占 (r, c), (r+1, c))
            if (r + 1 < 2*Quoridor::ROW_SIZE) {
                if (state.board[r][c] == 0 && state.board[r+1][c] == 0) {
                    Quoridor::State tmp = state;
                    bool ok = true;
                    for (int i = 0; i < Quoridor::WALL_LENGTH; i++) {
                        if (tmp.board[r + i][c]) { ok = false; break; }
                        tmp.board[r + i][c] = 1;
                    }
                    if (ok && isconnect(tmp.board, state.pos[0], 2*Quoridor::ROW_SIZE-1)
                         && isconnect(tmp.board, state.pos[1], 1))
                        result.push_back({r, c, 0});
                }
            }
            // 尝试水平墙 (占 (r, c), (r, c+1))
            if (c + 1 < 2*Quoridor::COLUMN_SIZE) {
                if (state.board[r][c] == 0 && state.board[r][c+1] == 0) {
                    Quoridor::State tmp = state;
                    bool ok = true;
                    for (int i = 0; i < Quoridor::WALL_LENGTH; i++) {
                        if (tmp.board[r][c + i]) { ok = false; break; }
                        tmp.board[r][c + i] = 1;
                    }
                    if (ok && isconnect(tmp.board, state.pos[0], 2*Quoridor::ROW_SIZE-1)
                         && isconnect(tmp.board, state.pos[1], 1))
                        result.push_back({r, c, 1});
                }
            }
        }
    }
    return result;
}

// ─── 最顶层：获取当前玩家的所有合法动作 ───
// 返回 list of Action，融合移动 + 放墙
static std::vector<Quoridor::Action> get_legal_actions(const Quoridor::State& state)
{
    std::vector<Quoridor::Action> result;

    // 移动
    auto moves = get_legal_moves_for_player(state, state.turn);
    for (auto& m : moves)
        result.emplace_back(m, false, 0);

    // 放墙
    auto walls = get_legal_walls_for_player(state, state.turn);
    for (auto& [r, c, d] : walls)
        result.emplace_back(std::pair<int,int>{r, c}, true, d);

    return result;
}


// =============================================================================
// pybind11 模块定义
// =============================================================================
PYBIND11_MODULE(quoridor_cpp, m)
{
    // ── 模块文档 ──
    m.doc() = "Quoridor C++ 游戏逻辑 — pybind11 绑定";

    // ── 常量 ──
    m.attr("ROW_SIZE")     = py::int_(Quoridor::ROW_SIZE);
    m.attr("COLUMN_SIZE")  = py::int_(Quoridor::COLUMN_SIZE);
    m.attr("WALL_NUM")     = py::int_(Quoridor::WALL_NUM);
    m.attr("WALL_LENGTH")  = py::int_(Quoridor::WALL_LENGTH);
    m.attr("ACTION_NUM")   = py::int_(Quoridor::ACTION_NUM);   // 81 移动 + 72 垂直墙 + 72 水平墙

    // ── Action 类 ──
    py::class_<Quoridor::Action>(m, "Action",
        R"(表示一步动作：移动或放墙。

        参数:
            pos: (row, col) — 移动目标位置 或 墙的起始坐标
            is_wall: True=放墙, False=移动
            wall_dir: 0=垂直墙, 1=水平墙 (仅放墙时有效)

        用法:
            act = Action((row, col), is_wall, wall_dir)
            ok = act.apply(state)       # 执行动作，成功返回 True
        )")
        .def(py::init<std::pair<int,int>, bool, int>(),
             py::arg("pos"), py::arg("is_wall"), py::arg("wall_dir"))
        .def_readwrite("pos",      &Quoridor::Action::pos)
        .def_readwrite("is_wall",  &Quoridor::Action::isWall)
        .def_readwrite("wall_dir", &Quoridor::Action::wall_dir)
        .def("apply", &Quoridor::Action::apply, py::arg("state"),
             R"(执行此动作。

            返回 True 表示成功应用，False 表示非法动作。
            成功时会修改 state 的 turn / pos / board。
            )")
        .def("__repr__", [](const Quoridor::Action& a) {
            if (a.isWall) {
                return "<Action: wall at (" + std::to_string(a.pos.first) + ","
                     + std::to_string(a.pos.second) + ") dir="
                     + std::to_string(a.wall_dir) + ">";
            } else {
                return "<Action: move to (" + std::to_string(a.pos.first) + ","
                     + std::to_string(a.pos.second) + ")>";
            }
        });

    // ── State 类 ──
    py::class_<Quoridor::State>(m, "State",
        R"(Quoridor 棋盘状态。

        用法:
            s = State()
            s.reset()
            print(s.turn)           # 当前轮到谁 (1 或 2)
            r, c = s.get_pos(1)     # 玩家 1 的位置
            cell = s.get_cell(r,c)  # 棋盘上的值 (0=空/通道, 1=墙)
        )")
        .def(py::init<>())
        .def("reset", &Quoridor::State::reset,
             "将棋盘重置为初始状态")

        // ── pos 读写 ──
        .def("get_pos", [](const Quoridor::State& s, int player) {
                return s.pos[player - 1];
            }, py::arg("player"),
            "获取某玩家位置 (player=1 或 2)，返回 (row, col)")
        .def("set_pos", [](Quoridor::State& s, int player, int r, int c) {
                s.pos[player - 1] = {r, c};
            }, py::arg("player"), py::arg("row"), py::arg("col"),
            "设置某玩家位置")

        // ── turn 只读 ──
        .def_property_readonly("turn", [](const Quoridor::State& s) {
                return s.turn;
            }, "当前轮到谁 (1 或 2)")
        .def("get_turn", [](const Quoridor::State& s) { return s.turn; })

        // ── wall_num 读写 ──
        .def("get_wall_num", [](const Quoridor::State& s, int player) {
                return s.wall_num[player - 1];
            }, py::arg("player"),
            "获取某玩家剩余墙数")
        .def("set_wall_num", [](Quoridor::State& s, int player, int n) {
                s.wall_num[player - 1] = n;
            }, py::arg("player"), py::arg("n"))

        // ── board 单元格读写 ──
        .def("get_cell", [](const Quoridor::State& s, int r, int c) {
                return s.board[r][c];
            }, py::arg("r"), py::arg("c"),
            "读取棋盘 cell (0=空, 1=墙)")
        .def("set_cell", [](Quoridor::State& s, int r, int c, bool v) {
                s.board[r][c] = v;
            }, py::arg("r"), py::arg("c"), py::arg("v"),
            "设置棋盘 cell")

        // ── 墙追踪读写 ──
        .def("get_h_wall", [](const Quoridor::State& s, int r, int c) {
                return s.h_wall[r][c];
            }, py::arg("r"), py::arg("c"),
            "读取水平墙标记")
        .def("get_v_wall", [](const Quoridor::State& s, int r, int c) {
                return s.v_wall[r][c];
            }, py::arg("r"), py::arg("c"),
            "读取垂直墙标记")

        // ── 深拷贝 (MCTS 树搜索时需要) ──
        .def("copy", [](const Quoridor::State& s) {
                return s;
            }, "返回状态的深拷贝副本")

        .def("__repr__", [](const Quoridor::State& s) {
            return "<State: turn=" + std::to_string(s.turn)
                 + " p1=(" + std::to_string(s.pos[0].first) + ","
                          + std::to_string(s.pos[0].second) + ")"
                 + " p2=(" + std::to_string(s.pos[1].first) + ","
                          + std::to_string(s.pos[1].second) + ")"
                 + " walls=" + std::to_string(s.wall_num[0]) + "/"
                            + std::to_string(s.wall_num[1])
                 + ">";
        });

    // ── 棋盘点位常量 (方便 Python 端引用) ──
    m.attr("ROW_MAX") = py::int_(2 * Quoridor::ROW_SIZE);     // 18
    m.attr("COL_MAX") = py::int_(2 * Quoridor::COLUMN_SIZE);  // 18
    m.attr("BOARD_SIZE") = py::int_(2 * Quoridor::ROW_SIZE + 1); // 19

    // ── 自由函数 ──

    // isconnect: 检查某玩家是否有通往目标行的路径
    m.def("isconnect", [](const Quoridor::State& state,
                          std::pair<int,int> start, int target_row) {
            return isconnect(state.board, start, target_row);
        }, py::arg("state"), py::arg("start"), py::arg("target_row"),
        R"(BFS 检查从 start 能否到达 target_row。

        用于放墙前校验是否阻挡了任意玩家的所有路径。
        player1 的目标行 = 18 (底部), player2 的目标行 = 1 (顶部)。
        )");

    // 获取某玩家所有合法移动目标
    m.def("get_legal_moves", &get_legal_moves_for_player,
        py::arg("state"), py::arg("player"),
        "获取某玩家的所有合法移动目标位置列表 [(r1,c1), (r2,c2), ...]");

    // 获取某玩家所有合法放墙位置
    m.def("get_legal_walls", &get_legal_walls_for_player,
        py::arg("state"), py::arg("player"),
        "获取某玩家的所有合法放墙位置列表 [(r,c,dir), ...]");

    // 获取当前玩家全部合法动作
    m.def("get_legal_actions", &get_legal_actions,
        py::arg("state"),
        "获取当前轮玩家的所有合法 Action 列表，融合移动 + 放墙");
}
