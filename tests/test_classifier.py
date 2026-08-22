import numpy as np
import pytest

from heard import classifier as clf


def fake_encode_factory(dim=16):
    """Deterministic encoder: each utterance gets a unit vector from its hash."""
    def encode(texts):
        out = np.empty((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = np.random.default_rng(abs(hash(t)) % (2**32)).standard_normal(dim)
            out[i] = h / np.linalg.norm(h)
        return out
    return encode


@pytest.fixture(scope="module")
def classifier():
    # Stable pseudo-embeddings: prototypes of one class cluster together.
    rng = np.random.default_rng(42)

    def encode(texts):
        rows = []
        for text in texts:
            tokens = set(text.lower().split())
            best_key, best_overlap = None, 0
            for cls_key, (_, _, utts) in clf.PROTOTYPES.items():
                overlap = max((len(tokens & set(u.split())) for u in utts), default=0)
                if overlap > best_overlap:
                    best_key, best_overlap = cls_key, overlap
            if best_key is None:
                vec = rng.standard_normal(32)
            else:
                seed = abs(hash(best_key)) % (2**32)
                vec = np.random.default_rng(seed).standard_normal(32)
                vec += rng.standard_normal(32) * 0.01
            rows.append(vec / np.linalg.norm(vec))
        return np.array(rows, dtype=np.float32)

    return clf.Classifier(encode)


class TestExtractPercent:
    @pytest.mark.parametrize("text,expected", [
        ("set volume to 50 percent", 50),
        ("set it to 73%", 73),
        ("volume to thirty percent", 30),
        ("make it sixty five", 65),
        ("half volume", 50),
        ("quarter volume", 25),
        ("set to one hundred percent", 100),
        ("no numbers here", None),
    ])
    def test_cases(self, text, expected):
        assert clf.extract_percent(text) == expected

    def test_out_of_range(self):
        assert clf.extract_percent("set volume to 400 percent") is None


class TestExtractWorkspace:
    @pytest.mark.parametrize("text,expected", [
        ("go to workspace 3", "3"),
        ("workspace seven", "7"),
        ("switch to workspace ten", "10"),
        ("nothing", None),
    ])
    def test_cases(self, text, expected):
        assert clf.extract_workspace(text) == expected


class TestExtractAppPhrase:
    def test_strips_verbs_and_filler(self):
        assert clf.extract_app_phrase("open the file manager") == "file manager"

    def test_keeps_core(self):
        assert clf.extract_app_phrase("spotify") == "spotify"


class TestClassifierVerdicts:
    def test_exact_prototype_classifies(self, classifier):
        v = classifier.match("how's my battery")
        assert v.tool_call == {"name": "system_query",
                               "arguments": {"query": "battery"}}
        assert not v.declined

    def test_action_from_class(self, classifier):
        v = classifier.match("skip this one")
        assert v.tool_call["name"] == "media_control"
        assert v.tool_call["arguments"]["action"] == "next"

    def test_volume_set_needs_number(self, classifier):
        v = classifier.match("set volume to fifty percent")
        assert v.tool_call is not None
        assert v.tool_call["arguments"]["amount"] == "50"

    def test_volume_set_without_number_falls_back(self, classifier):
        v = classifier.match("set volume please")   # no number in transcript
        assert v.tool_call is None                  # must not dispatch without amount
        assert isinstance(v, clf.Verdict)

    def test_workspace_slot(self, classifier):
        v = classifier.match("go to workspace three")
        assert v.tool_call == {"name": "workspace_switch",
                               "arguments": {"workspace": "3"}}

    def test_focus_requires_fallback(self, classifier):
        # focus has no prototypes and is gated -> can never be fast-accepted
        assert clf.PROTOTYPES["window_action:focus"][2] == []
        assert "window_action:focus" in clf.FALLBACK_CLASSES
        v = classifier.match("close this window")
        assert v.tool_call["name"] == "window_action"
        assert v.tool_call["arguments"]["action"] == "close"

    def test_unknown_off_topic_declined_or_fallback(self, classifier):
        v = classifier.match("what is the weather tomorrow")
        assert v.tool_call is None   # never dispatches garbage

    def test_verdict_defaults(self):
        v = clf.Verdict()
        assert v.tool_call is None and not v.declined and v.score is None
