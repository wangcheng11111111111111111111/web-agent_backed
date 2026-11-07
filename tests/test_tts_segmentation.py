from __future__ import annotations

import unittest

from app.services.audio import (
    MIN_TTS_SPOKEN_CHARS,
    _group_tts_paragraphs,
    _normalize_spoken_text,
    _prepare_cosyvoice_text,
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
        self.assertTrue(
            all(
                _spoken_char_count(group) >= MIN_TTS_SPOKEN_CHARS
                or index == len(groups) - 1
                for index, group in enumerate(groups)
            )
        )

    def test_short_fragments_do_not_force_a_decreasing_one_char_chain(self) -> None:
        text = "开头。\n\n今天谈。\n\n人工智能。\n\n与。\n\n未来。"

        groups = _group_tts_paragraphs(text)

        self.assertNotIn("与。", groups)
        self.assertGreaterEqual(_spoken_char_count(groups[0]), 8)

    def test_long_sentence_is_split_only_near_a_natural_boundary(self) -> None:
        text = (
            "这是一个很长的句子，用来验证在超过目标长度时，"
            "分段器会优先选择自然标点附近的位置，而不会拆成单个字符。"
        )

        segments = _split_tts_sentences(text, max_sentence_chars=30)

        self.assertGreater(len(segments), 1)
        self.assertTrue(all(_spoken_char_count(item) > 1 for item in segments))

    def test_cosyvoice_input_is_normalized_without_secondary_split(self) -> None:
        class Frontend:
            def __init__(self) -> None:
                self.arguments: tuple[bool, bool] | None = None

            def text_normalize(
                self,
                text: str,
                *,
                split: bool,
                text_frontend: bool,
            ) -> str:
                self.arguments = (split, text_frontend)
                return f"已规范化：{text}"

        model = type("Model", (), {})()
        model.frontend = Frontend()

        prepared = _prepare_cosyvoice_text(model, "测试文本")

        self.assertEqual(prepared, "已规范化：测试文本")
        self.assertEqual(model.frontend.arguments, (False, True))


if __name__ == "__main__":
    unittest.main()
