from __future__ import annotations

import unittest

from app.services.audio import (
    TTS_FIRST_SEGMENT_MAX_CHARS,
    TTS_FOLLOWING_SEGMENT_MAX_CHARS,
    TTS_MIN_SEGMENT_CHARS,
    TTS_MIN_TRAILING_CHARS,
    _group_tts_paragraphs,
    _normalize_spoken_text,
    _split_tts_sentences,
    _spoken_char_count,
)


class TTSSegmentationTests(unittest.TestCase):
    def test_short_paragraphs_are_merged_into_a_safe_tts_chunk(self) -> None:
        text = (
            "各位好。\n\n"
            "今天谈。\n\n"
            "AI。\n\n"
            "与。\n\n"
            "未来。\n\n"
            "工作。"
        )

        groups = _group_tts_paragraphs(text)

        self.assertEqual(groups, ["各位好。今天谈。AI。与。未来。工作。"])

    def test_short_fragments_do_not_force_a_decreasing_one_char_chain(self) -> None:
        text = "开头。\n\n今天谈。\n\n人工智能。\n\n与。\n\n未来。"

        groups = _group_tts_paragraphs(text)

        self.assertNotIn("与。", groups)
        self.assertEqual(groups, ["开头。今天谈。人工智能。与。未来。"])

    def test_first_chunk_is_larger_and_following_chunks_are_bounded(self) -> None:
        first = "甲" * 118 + "。"
        second = "乙" * 30 + "。"
        third = "丙" * 30 + "。"

        groups = _group_tts_paragraphs(first + second + third)
        lengths = [_spoken_char_count(group) for group in groups]

        self.assertGreater(len(groups), 1)
        self.assertGreater(lengths[0], lengths[1])
        self.assertLessEqual(lengths[0], TTS_FIRST_SEGMENT_MAX_CHARS)
        self.assertTrue(
            all(
                length <= TTS_FOLLOWING_SEGMENT_MAX_CHARS
                for length in lengths[1:]
            )
        )

    def test_long_sentence_carries_remainder_at_natural_punctuation(self) -> None:
        text = "甲" * 35 + "，" + "乙" * 35 + "，" + "丙" * 35 + "。"

        segments = _split_tts_sentences(text, max_sentence_chars=60)

        self.assertGreater(len(segments), 1)
        self.assertTrue(all(len(item) <= 60 for item in segments))
        self.assertTrue(all(item.rstrip().endswith("，") for item in segments[:-1]))
        self.assertTrue(
            all(
                _spoken_char_count(item) >= TTS_MIN_TRAILING_CHARS
                for item in segments
            )
        )

    def test_short_tail_is_not_emitted_as_a_tiny_chunk(self) -> None:
        text = "甲" * 100 + "。" + "这是结尾。"

        groups = _group_tts_paragraphs(text)

        self.assertEqual(len(groups), 1)
        self.assertGreaterEqual(
            _spoken_char_count(groups[0]),
            TTS_MIN_SEGMENT_CHARS,
        )

    def test_slightly_over_limit_sentence_avoids_tiny_remainder(self) -> None:
        text = "甲" * 90 + "。"

        segments = _split_tts_sentences(
            text,
            max_sentence_chars=TTS_FOLLOWING_SEGMENT_MAX_CHARS,
        )

        self.assertEqual(len(segments), 2)
        self.assertTrue(
            all(
                TTS_MIN_TRAILING_CHARS
                <= _spoken_char_count(segment)
                <= TTS_FOLLOWING_SEGMENT_MAX_CHARS
                for segment in segments
            )
        )

    def test_english_identifier_is_not_transliterated(self) -> None:
        normalized = _normalize_spoken_text(
            "m0NESY 获得 MVP，NiKo 终于夺冠。"
        )

        self.assertIn("m zero NESY", normalized)
        self.assertIn("NiKo", normalized)
        self.assertNotIn("莫内西", normalized)
        self.assertNotIn("尼科", normalized)

    def test_markdown_bold_preserves_english_name(self) -> None:
        normalized = _normalize_spoken_text(
            "**karrigan 创纪录**：36 岁 2 个月"
        )

        self.assertEqual(normalized, "karrigan 创纪录：36 岁 2 个月")
        self.assertNotIn("*", normalized)

    def test_alphanumeric_identifiers_are_protected_generically(self) -> None:
        normalized = _normalize_spoken_text(
            "账号 x7K9、模型 GPT-4.1 和工具 v2 都要保留英文数字读法。"
        )

        self.assertIn("x seven K nine", normalized)
        self.assertIn("GPT four point one", normalized)
        self.assertIn("v two", normalized)
        self.assertNotIn("七", normalized)
        self.assertNotIn("四", normalized)


if __name__ == "__main__":
    unittest.main()
