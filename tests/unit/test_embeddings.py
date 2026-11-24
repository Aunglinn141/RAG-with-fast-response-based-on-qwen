"""BGE adapter contracts without loading weights or contacting Ollama."""

import unittest
from unittest.mock import patch

from local_rag.clients.ollama import OllamaError
from local_rag.core.config import VectorSettings
from local_rag.embeddings.bge import BGEEmbeddings


def vector(value=1.0):
    return [value] + [0.0] * 1023


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.provider = BGEEmbeddings(VectorSettings(batch_size=2))

    def test_documents_preserve_order_across_bounded_batches(self):
        texts = ["第一块", "第二块", "第三块", "第四块", "第五块"]
        expected = [vector(i + 1) for i in range(len(texts))]
        lookup = dict(zip(texts, expected))
        with patch.object(self.provider.client, "embed",
                          side_effect=lambda model, batch, **kw: [lookup[t] for t in batch]) as embed:
            actual = self.provider.embed_documents(texts)
        self.assertEqual(actual, expected)
        self.assertEqual([call.args for call in embed.call_args_list], [
            ("bge-m3", texts[:2]), ("bge-m3", texts[2:4]), ("bge-m3", texts[4:]),
        ])
        self.assertTrue(all(call.kwargs == {"keep_alive": "-1"} for call in embed.call_args_list))
        self.assertEqual(texts, ["第一块", "第二块", "第三块", "第四块", "第五块"])

    def test_query_returns_one_vector_without_instruction_prefix(self):
        query = "图书馆什么时候开放？"
        with patch.object(self.provider.client, "embed", return_value=[vector()]) as embed:
            result = self.provider.embed_query(query)
        self.assertEqual(result, vector())
        embed.assert_called_once_with("bge-m3", [query], keep_alive="-1")

    def test_empty_document_list_needs_no_model(self):
        with patch.object(self.provider.client, "embed") as embed:
            self.assertEqual(self.provider.embed_documents([]), [])
            embed.assert_not_called()

    def test_invalid_document_batch_fails_before_any_inference(self):
        for value in ("not a list", None, ("tuple",), ["valid", "valid", "  "],
                      ["valid", None], [123], [""]):
            with self.subTest(value=value), patch.object(self.provider.client, "embed") as embed:
                with self.assertRaises(OllamaError):
                    self.provider.embed_documents(value)
                embed.assert_not_called()

    def test_query_rejects_blank_and_non_string_values(self):
        for value in ("", " \n\t", None, ["question"], 123, True):
            with self.subTest(value=value), patch.object(self.provider.client, "embed") as embed:
                with self.assertRaises(OllamaError):
                    self.provider.embed_query(value)
                embed.assert_not_called()

    def test_invalid_vectors_are_rejected_by_both_entry_points(self):
        for values in ([], [[1.0]], [vector(0)], [vector(float("nan"))],
                       [vector(float("inf"))], [vector(True)], [vector(), vector()]):
            for method, value in ((self.provider.embed_documents, ["text"]),
                                  (self.provider.embed_query, "question")):
                with self.subTest(values=values, method=method.__name__):
                    with patch.object(self.provider.client, "embed", return_value=values):
                        with self.assertRaises(OllamaError):
                            method(value)

    def test_failed_later_batch_does_not_return_partial_results(self):
        with patch.object(self.provider.client, "embed",
                          side_effect=[[vector(), vector()], OllamaError("offline")]) as embed:
            with self.assertRaisesRegex(OllamaError, "offline"):
                self.provider.embed_documents(["one", "two", "three", "four", "five"])
            self.assertEqual(embed.call_count, 2)

    def test_embedding_does_not_call_chat(self):
        with patch.object(self.provider.client, "chat", side_effect=AssertionError("chat called")):
            with patch.object(self.provider.client, "embed", return_value=[vector()]):
                self.provider.embed_documents(["document"])
                self.provider.embed_query("question")

    def test_batch_size_requires_integer(self):
        for size in (0, -1, 257, True, 1.5, "2"):
            with self.subTest(size=size), self.assertRaises(ValueError):
                VectorSettings(batch_size=size)

    def test_vector_timeouts_must_be_finite_and_positive(self):
        for name in ("timeout", "lock_timeout"):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    VectorSettings(**{name: value})

    def test_chat_model_configuration_cannot_replace_bge(self):
        with self.assertRaises(ValueError):
            VectorSettings(model_name="qwen3:1.7b")


if __name__ == "__main__":
    unittest.main()
