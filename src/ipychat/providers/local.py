# -*- coding: utf-8 -*-

from typing import Generator

import ollama

from .base import BaseProvider
import contextlib


class OllamaProvider(BaseProvider):
    def initialize_client(self) -> None:
        self.model = self.config["current"]["model"]
        self.client = ollama.Client()
        # todo options like temperate ctx etc

    def stream_chat(
        self, system_prompt: str, user_content: str
    ) -> Generator[str, None, None]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

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
