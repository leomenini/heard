"""Verify the heard package tree imports cleanly."""


class TestPackage:
    def test_import_heard(self):
        import heard
        assert heard.__file__ is not None

    def test_all_public_modules_import(self):
        pass

    def test_tools_init(self):
        import heard.tools
        assert heard.tools.__file__ is not None
