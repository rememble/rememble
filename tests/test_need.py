"""Tests for memory need analysis."""

from __future__ import annotations

from rememble.search.need import analyzeMemoryNeed


class TestSkipPatterns:
    def test_empty(self):
        result = analyzeMemoryNeed("")
        assert not result.should_recall
        assert result.need_type == "none"

    def test_whitespace(self):
        result = analyzeMemoryNeed("   ")
        assert not result.should_recall

    def test_acknowledgements(self):
        for q in ["ok", "thanks", "got it", "sure", "yep", "nice!", "lgtm"]:
            result = analyzeMemoryNeed(q)
            assert not result.should_recall, f"Expected skip for '{q}'"
            assert result.need_type == "none"

    def test_shortQuery(self):
        result = analyzeMemoryNeed("run tests")
        assert not result.should_recall
        assert result.need_type == "none"


class TestTemporalPatterns:
    def test_whatChanged(self):
        result = analyzeMemoryNeed("what changed since last week?")
        assert result.should_recall
        assert result.need_type == "temporal"

    def test_recently(self):
        result = analyzeMemoryNeed("what did we do recently?")
        assert result.should_recall
        assert result.need_type == "temporal"

    def test_lastTime(self):
        result = analyzeMemoryNeed("what happened last time?")
        assert result.should_recall
        assert result.need_type == "temporal"


class TestIdentityPatterns:
    def test_whoAmI(self):
        result = analyzeMemoryNeed("who am I?")
        assert result.should_recall
        assert result.need_type == "identity"

    def test_myPreferences(self):
        result = analyzeMemoryNeed("what are my preferences?")
        assert result.should_recall
        assert result.need_type == "identity"


class TestOpenLoopPatterns:
    def test_nextSteps(self):
        result = analyzeMemoryNeed("what are the next steps?")
        assert result.should_recall
        assert result.need_type == "open_loop"

    def test_didWeDecide(self):
        result = analyzeMemoryNeed("did we decide on the approach?")
        assert result.should_recall
        assert result.need_type == "open_loop"

    def test_followUp(self):
        result = analyzeMemoryNeed("where did we leave off on the migration?")
        assert result.should_recall
        assert result.need_type == "open_loop"


class TestBroadContextPatterns:
    def test_catchMeUp(self):
        result = analyzeMemoryNeed("catch me up on the project")
        assert result.should_recall
        assert result.need_type == "broad_context"

    def test_recap(self):
        result = analyzeMemoryNeed("give me a recap")
        assert result.should_recall
        assert result.need_type == "broad_context"

    def test_summary(self):
        result = analyzeMemoryNeed("summary of what we know")
        assert result.should_recall
        assert result.need_type == "broad_context"


class TestProspectivePatterns:
    def test_remindMe(self):
        result = analyzeMemoryNeed("remind me to update the docs")
        assert result.should_recall
        assert result.need_type == "prospective"

    def test_dontForget(self):
        result = analyzeMemoryNeed("don't forget to run the migration")
        assert result.should_recall
        assert result.need_type == "prospective"


class TestFactLookupPatterns:
    def test_whatIs(self):
        result = analyzeMemoryNeed("what is the deployment process?")
        assert result.should_recall
        assert result.need_type == "fact_lookup"

    def test_whoQuestion(self):
        result = analyzeMemoryNeed("who is responsible for the auth service?")
        assert result.should_recall
        assert result.need_type == "fact_lookup"


class TestDefaultRecall:
    def test_normalQueryRecalls(self):
        result = analyzeMemoryNeed("the database migration strategy for user accounts")
        assert result.should_recall
        assert result.need_type == "general"
        assert result.confidence == 0.5

    def test_codeQueryRecalls(self):
        result = analyzeMemoryNeed("how does the authentication middleware work in this project")
        assert result.should_recall


class TestConfidence:
    def test_emptyHighConfidence(self):
        result = analyzeMemoryNeed("")
        assert result.confidence == 1.0

    def test_ackHighConfidence(self):
        result = analyzeMemoryNeed("ok")
        assert result.confidence >= 0.9

    def test_defaultLowConfidence(self):
        result = analyzeMemoryNeed("something about the codebase architecture and patterns")
        assert result.confidence <= 0.6
