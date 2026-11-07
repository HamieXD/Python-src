import json

from config import Settings, load_settings, save_settings


def test_missing_settings_file_uses_defaults(tmp_path):
    path = tmp_path / "settings.json"

    assert load_settings(path) == Settings()


def test_broken_settings_file_uses_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{broken", encoding="utf-8")

    assert load_settings(path) == Settings()


def test_unknown_settings_keys_are_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"camera_index": 2, "future_option": True}),
        encoding="utf-8",
    )

    assert load_settings(path) == Settings(camera_index=2)


def test_invalid_setting_uses_only_its_default_and_warns(tmp_path, capsys):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"camera_index": 2, "alarm_volume": 1.7}),
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.camera_index == 2
    assert settings.alarm_volume == Settings().alarm_volume
    assert "alarm_volume" in capsys.readouterr().out


def test_save_settings_writes_values_that_can_be_loaded(tmp_path):
    path = tmp_path / "settings.json"
    settings = Settings(
        camera_index=1,
        ear_threshold=0.27,
        eyes_closed_seconds=1.2,
        face_lost_warning_seconds=2.5,
        face_lost_alarm_delay_seconds=3.0,
        alarm_volume=0.6,
    )

    save_settings(settings, path)

    assert load_settings(path) == settings


def test_detection_settings_have_defaults_and_validate_independently(tmp_path, capsys):
    defaults = load_settings(tmp_path / "missing.json")
    assert getattr(defaults, "calibration_seconds", None) == 5.0
    assert getattr(defaults, "yawn_mar_threshold", None) == 0.6
    assert getattr(defaults, "head_nod_drop_ratio", None) == 0.15

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "calibration_seconds": 8,
                "yawn_mar_threshold": -0.2,
                "head_nod_drop_ratio": 0.25,
            }
        ),
        encoding="utf-8",
    )
    settings = load_settings(path)

    assert settings.calibration_seconds == 8
    assert settings.yawn_mar_threshold == 0.6
    assert settings.head_nod_drop_ratio == 0.25
    assert "yawn_mar_threshold" in capsys.readouterr().out


def test_unrepresentable_numeric_setting_falls_back_without_discarding_others(
    tmp_path, capsys
):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"ear_threshold": 10**400, "camera_index": 3}),
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.camera_index == 3
    assert settings.ear_threshold == Settings().ear_threshold
    assert "ear_threshold" in capsys.readouterr().out
