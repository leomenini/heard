from unittest import mock

from heard.tools.helpers.apps import app_exists


class TestAppExists:
    @mock.patch("heard.tools.helpers.apps.shutil.which", return_value="/usr/bin/firefox")
    def test_found(self, mock_which):
        assert app_exists("firefox") is True
        mock_which.assert_called_once_with("firefox")

    @mock.patch("heard.tools.helpers.apps.shutil.which", return_value=None)
    def test_not_found(self, mock_which):
        assert app_exists("nonexistent") is False


class TestLocalizedNames:
    def test_desktop_entry_indexes_localized_names(self, tmp_path, monkeypatch):
        from heard.tools.helpers import apps

        apps_dir = tmp_path / "applications"
        apps_dir.mkdir()
        (apps_dir / "calc.desktop").write_text(
            "[Desktop Entry]\n"
            "Name=Calculator\n"
            "Name[es]=Calculadora\n"
            "GenericName=Math tool\n"
            "Exec=gnome-calculator\n"
        )
        monkeypatch.setattr(apps, "_desktop_dirs", lambda: [apps_dir])
        apps.installed_apps.cache_clear()
        try:
            table = apps.installed_apps()
            assert table["calculadora"] == "gnome-calculator"
            assert table["calculator"] == "gnome-calculator"
            assert table["math tool"] == "gnome-calculator"
        finally:
            apps.installed_apps.cache_clear()

    def test_hidden_entry_skipped(self, tmp_path, monkeypatch):
        from heard.tools.helpers import apps

        apps_dir = tmp_path / "applications"
        apps_dir.mkdir()
        (apps_dir / "ghost.desktop").write_text(
            "[Desktop Entry]\nName=Ghost\nNoDisplay=true\nExec=ghost\n"
        )
        monkeypatch.setattr(apps, "_desktop_dirs", lambda: [apps_dir])
        apps.installed_apps.cache_clear()
        try:
            assert "ghost" not in apps.installed_apps()
        finally:
            apps.installed_apps.cache_clear()
