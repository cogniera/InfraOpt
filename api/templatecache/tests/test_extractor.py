"""Tests for _format_compound_list_response in utils/extractor.py."""

import pytest

from templatecache.utils.extractor import _format_compound_list_response


# Shared test response — parenthetical list format
_RESPONSE = (
    "Eight planets orbiting the Sun: Mercury (smallest, closest), "
    "Venus (hottest, thick atmosphere), Earth (life), Mars (red, rovers), "
    "Jupiter (largest, Great Red Spot), Saturn (rings), Uranus (tilted axis), "
    "Neptune (farthest, windiest). Pluto was reclassified as a dwarf planet in 2006."
)


class TestCompoundFormatterFires:
    """Compound query with superlative + list + conjunction returns non-None."""

    def test_superlative_then_list(self):
        """'name the largest ... then list the rest' triggers compound formatter."""
        result = _format_compound_list_response(
            "name the largest planet in the solar system then list the rest of the planets",
            _RESPONSE,
        )
        assert result is not None
        assert "Jupiter" in result

    def test_smallest_then_list(self):
        """'what is the smallest ... then list all' triggers compound formatter."""
        result = _format_compound_list_response(
            "what is the smallest planet then list all planets",
            _RESPONSE,
        )
        assert result is not None
        assert "Mercury" in result


class TestSingleFactualReturnsNone:
    """Single factual query without list signal returns None."""

    def test_single_superlative(self):
        """'what is the largest planet' has no list signal or compound."""
        result = _format_compound_list_response(
            "what is the largest planet in the solar system",
            _RESPONSE,
        )
        assert result is None

    def test_single_superlative_no_conjunction(self):
        """'name the biggest planet' has no compound conjunction."""
        result = _format_compound_list_response(
            "name the biggest planet",
            _RESPONSE,
        )
        assert result is None


class TestListOnlyReturnsNone:
    """List-only query with no superlative returns None."""

    def test_list_all(self):
        """'list all the planets' has no superlative."""
        result = _format_compound_list_response(
            "list all the planets in the solar system",
            _RESPONSE,
        )
        assert result is None

    def test_name_every(self):
        """'name every planet' has no superlative."""
        result = _format_compound_list_response(
            "name every planet",
            _RESPONSE,
        )
        assert result is None


class TestExtractionFailsReturnsNone:
    """Compound query where extraction fails returns None."""

    def test_no_matching_item(self):
        """Superlative 'brightest' has no match in response descriptors."""
        result = _format_compound_list_response(
            "name the brightest planet then list the rest",
            _RESPONSE,
        )
        assert result is None


class TestOutputFormat:
    """Extracted answer appears before full list with separator."""

    def test_answer_before_list(self):
        """Extracted answer is the first content in the output."""
        result = _format_compound_list_response(
            "name the largest planet then list the rest",
            _RESPONSE,
        )
        assert result is not None
        # Answer must come before the full list
        jupiter_pos = result.index("Jupiter")
        list_pos = result.index("Eight planets")
        assert jupiter_pos < list_pos

    def test_full_list_separator(self):
        """'Full list:' appears as a separator between answer and list."""
        result = _format_compound_list_response(
            "name the largest planet then list the rest",
            _RESPONSE,
        )
        assert result is not None
        assert "Full list:" in result

    def test_full_response_preserved(self):
        """The complete original response appears after the separator."""
        result = _format_compound_list_response(
            "name the largest planet then list the rest",
            _RESPONSE,
        )
        assert result is not None
        after_separator = result.split("Full list:\n", 1)[1]
        assert after_separator == _RESPONSE

