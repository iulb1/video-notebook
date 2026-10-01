from __future__ import annotations

import unittest

import web_app


class WebConfigTests(unittest.TestCase):
    def setUp(self):
        self.original_queue = list(web_app.task_queue)
        web_app.task_queue.clear()

    def tearDown(self):
        web_app.task_queue[:] = self.original_queue

    def test_request_without_backend_uses_sensevoice(self):
        config = web_app.build_config({"input": "a", "output": "b"})
        self.assertEqual(config["backend"], "sensevoice")
        self.assertEqual(config["model"], "iic/SenseVoiceSmall")
        self.assertFalse(config["trash_source"])

    def test_queue_rejects_different_model(self):
        first = web_app.build_config({"backend": "faster-whisper"})
        web_app.task_queue.append({"config": first})
        second = web_app.build_config({"backend": "sensevoice"})
        self.assertTrue(web_app.queue_model_conflict(second))
        self.assertFalse(web_app.queue_model_conflict(first))


if __name__ == "__main__":
    unittest.main()
