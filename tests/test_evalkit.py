import re
from csat_forge import evalkit as ek


def test_parse_choice():
    assert ek.parse_choice("풀이...\n정답: 3") == 3
    assert ek.parse_choice("정답 : **④**") == 4
    assert ek.parse_choice("답은 2번입니다") == 2
    assert ek.parse_choice("정답: 7") is None
    assert ek.parse_choice("") is None


def test_wilson_bounds():
    lo, hi = ek.wilson(50, 100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61


class Oracle:
    """Always right → accuracy 100%, no misconception stats."""
    def complete(self, prompt):
        raise AssertionError


def _items(n=60):
    return ek.draw_items(n, seed=7)


def test_random_has_no_attraction():
    items = ek.draw_items(300, seed=11)
    res = ek.run_eval(ek.RandomBackend(3), items)
    a = ek.analyze(res)
    assert abs(a["accuracy"] - 0.2) < 0.08
    assert 0.7 < a["overall_lift"] < 1.3


def test_trap_lover_detected():
    """A model that always picks a tagged trap when it is wrong must show lift >> 1."""
    items = ek.draw_items(150, seed=5)
    res = []
    for rec in items:
        tm = rec["completion"]["trap_map"]
        pick = int(sorted(tm)[0])
        cidx = ek.correct_index(rec)
        res.append({"generator": rec["generator"], "topic": rec["spec"]["topic"],
                    "picked": pick, "correct": cidx, "ok": pick == cidx, "trap_options": tm,
                    "n_choices": len(rec["completion"]["choices"]), "trap_picked": tm[str(pick)]})
    a = ek.analyze(res)
    assert a["overall_lift"] > 2.0


def test_items_are_fresh_and_reproducible():
    a = [r["completion"]["question"] for r in ek.draw_items(20, seed=1)]
    b = [r["completion"]["question"] for r in ek.draw_items(20, seed=1)]
    c = [r["completion"]["question"] for r in ek.draw_items(20, seed=2)]
    assert a == b and a != c
