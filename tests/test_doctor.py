"""doctor 的判据测试。

重点在 classify_model_probe：它的结论直接决定「去控制台开通」还是
「去改配置」。判错了人会白跑一趟，所以每种组合都要钉住。
"""

from __future__ import annotations

import json

from avatar_tryon.doctor import (
    BLOCK,
    OK,
    WARN,
    Check,
    Report,
    check_python,
    classify_model_probe,
    render,
    run_checks,
    to_json,
)

NOT_EXIST = '{"code":"InvalidParameter","message":"Model not exist.","request_id":"x"}'
PARAM_ERR = '{"code":"InvalidParameter","message":"input.person_image_url is required"}'


def test_probe_404_same_message_means_not_enabled():
    """404 + Model not exist：模型名对，账号没开通。这是实测到的真实组合。"""
    verdict, why = classify_model_probe(404, NOT_EXIST)
    assert verdict == "not_enabled"
    assert "开通" in why


def test_probe_400_same_message_means_wrong_name():
    """400 + Model not exist：名字不存在。与 404 同样文案，结论必须不同。"""
    verdict, why = classify_model_probe(400, NOT_EXIST)
    assert verdict == "unknown_name"
    assert "改名" in why or "不存在" in why


def test_probe_param_error_means_model_is_available():
    """参数校验报错说明模型名已被解析——模型本身通。"""
    verdict, _ = classify_model_probe(400, PARAM_ERR)
    assert verdict == "available"


def test_probe_200_is_available():
    assert classify_model_probe(200, "{}")[0] == "available"


def test_probe_5xx_is_inconclusive():
    assert classify_model_probe(502, "bad gateway")[0] == "inconclusive"


def test_python_312_is_ok_for_real_engine():
    check = check_python((3, 12, 9))
    assert check.status == OK


def test_python_314_blocks_real_engine_and_says_why():
    check = check_python((3, 14, 5))
    assert check.status == BLOCK
    assert "3.12" in check.detail
    assert ".venv-lhm" in check.action


def test_tier_summary_keeps_all_ok_tier():
    """回归：早先的实现只记录「非 ok」的档位，全 ok 的档位会整行消失。"""
    report = Report()
    report.add(Check("a", OK, "", tier="proxy"))
    report.add(Check("b", BLOCK, "", tier="real"))
    tiers = report.by_tier()
    assert tiers["proxy"] == OK
    assert tiers["real"] == BLOCK


def test_tier_summary_takes_worst_status():
    report = Report()
    report.add(Check("a", OK, "", tier="real"))
    report.add(Check("b", WARN, "", tier="real"))
    assert report.by_tier()["real"] == WARN


def test_render_shows_action_only_for_non_ok():
    report = Report()
    report.add(Check("好的", OK, "没问题", action="不该显示"))
    report.add(Check("坏的", WARN, "缺东西", action="去补"))
    text = render(report)
    assert "不该显示" not in text
    assert "-> 去补" in text


def test_offline_run_does_not_touch_network():
    """offline 模式只做本地检查；这条测试保证 CI 不会因为 doctor 而联网。"""
    report = run_checks(root=__import__("pathlib").Path("."), offline=True)
    names = [c.name for c in report.checks]
    assert any("Python" in n for n in names)
    assert any("联网探测" in n for n in names)
    # 联网项在 offline 下不应出现
    assert not any("aitryon" in n for n in names)


def test_json_output_is_parseable_and_has_tiers():
    report = Report()
    report.add(Check("a", OK, "d", tier="proxy"))
    data = json.loads(to_json(report))
    assert data["tiers"]["proxy"] == OK
    assert data["checks"][0]["name"] == "a"
