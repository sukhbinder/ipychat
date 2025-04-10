# -*- coding: utf-8 -*-

from typing import Generator

import ollama

from .base import BaseProvider
import contextlib


class OllamaProvider(BaseProvider):
    def initialize_client(self) -> None:
        self.model = self.config["current"]["model"]
        self.client = ollama.Client()
        self.temperature = self.config.get("local", {}).get("temperature", None)
        self.num_ctx = self.config.get("local", {}).get("num_ctx", None)

    def stream_chat(
        self, system_prompt: str, user_content: str
    ) -> Generator[str, None, None]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        options = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
            )

            for chunk in response:
                with contextlib.suppress(KeyError):
                    yield chunk["message"]["content"]
        except ollama.ResponseError as ex:
            print(ex)
            pass
