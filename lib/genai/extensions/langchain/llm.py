"""Wrapper around IBM GENAI APIs for use in Langchain"""
import asyncio
import logging
import re
from functools import partial
from typing import Any, Iterator, List, Mapping, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.schema import LLMResult
from langchain.schema.output import Generation, GenerationChunk
from pydantic import BaseModel, Extra

from genai.exceptions import GenAiException

try:
    from langchain.llms.base import LLM
except ImportError:
    raise ImportError("Could not import langchain: Please install ibm-generative-ai[langchain] extension.")

from genai import Credentials, Model
from genai.schemas import GenerateParams

logger = logging.getLogger(__name__)

__all__ = ["LangChainInterface"]


class LangChainInterface(LLM, BaseModel):
    """
    Wrapper around IBM GENAI models.
    To use, you should have the ``genai`` python package installed
    and initialize the credentials attribute of this class with
    an instance of ``genai.Credentials``. Model specific parameters
    can be passed through to the constructor using the ``params``
    parameter, which is an instance of GenerateParams.
    Example:
        .. code-block:: python
            llm = LangChainInterface(model="google/flan-ul2", credentials=creds)
    """

    credentials: Credentials = None
    model: Optional[str] = None
    params: Optional[GenerateParams] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _params = self._get_params()
        return {
            **{"model": self.model},
            **{"params": _params},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "IBM GENAI"

    def _get_params(self):
        if self.params is None:
            return GenerateParams()

        if isinstance(self.params, dict):
            return GenerateParams(**self.params)
        return self.params.copy()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the IBM GENAI's inference endpoint.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                llm = LangChainInterface(model_id="google/flan-ul2", credentials=creds)
                response = llm("What is a molecule")
        """
        result = self._generate(prompts=[prompt], stop=stop, run_manager=run_manager, **kwargs)
        return result.generations[0][0].text

    def _update_llm_result(self, current: LLMResult, generation_info: Optional[dict]):
        if generation_info is None:
            return

        token_usage = current.llm_output["token_usage"]
        for key in {"generated_token_count", "input_token_count"}:
            if key in generation_info:
                new_tokens = generation_info.get(key, 0) or 0
                token_usage[key] = token_usage.get(key, 0) + new_tokens

    def _create_generation_info(self, response: dict):
        keys_to_pick = {"stop_reason"}
        return {k: v for k, v in response.items() if k in keys_to_pick and v is not None}

    def _create_full_generation_info(self, response: dict):
        keys_to_omit = {"generated_text"}
        return {k: v for k, v in response.items() if k not in keys_to_omit and v is not None}

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        print(f"generating for prompt {prompts}")
        result = LLMResult(generations=[], llm_output={"token_usage": {}})
        if len(prompts) == 0:
            return result

        params = self._get_params()
        params.stop_sequences = stop or params.stop_sequences
        if params.stream:
            if len(prompts) != 1:
                raise GenAiException(ValueError("Streaming works only for a single prompt."))

            generation = GenerationChunk(text="", generation_info={})
            for chunk in self._stream(
                prompt=prompts[0],
                stop=params.stop_sequences,
                run_manager=run_manager,
                **kwargs,
            ):
                self._update_llm_result(current=result, generation_info=chunk.generation_info)
                generation.text += chunk.text
                generation.generation_info.update(self._create_generation_info(chunk.generation_info or {}))

            result.generations.append([generation])
            return result

        model = Model(model=self.model, params=params, credentials=self.credentials)
        for response in model.generate(
            prompts=prompts,
            **kwargs,
        ):
            if params.stop_sequences:
                response.generated_text = self._enforce_stop_tokens(response.generated_text, params.stop_sequences)

            generation = Generation(
                text=response.generated_text or "",
                generation_info=self._create_full_generation_info(response.dict()),
            )
            logger.info("Output of GENAI call: {}".format(generation.text))
            result.generations.append([generation])
            self._update_llm_result(current=result, generation_info=generation.generation_info)

        return result

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self._generate, prompts, stop, run_manager, **kwargs)
        )

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Call the IBM GENAI's inference endpoint which then streams the response.
        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager.
        Returns:
            The iterator which yields generation chunks.
        Example:
            .. code-block:: python
                llm = LangChainInterface(model_id="google/flan-ul2", credentials=creds)
                for chunk in llm.stream("What is a molecule?"):
                    print(chunk.text)
        """
        params = self._get_params()
        params.stop_sequences = stop or params.stop_sequences

        model = Model(model=self.model, params=params, credentials=self.credentials)
        for response in model.generate_stream(prompts=[prompt], **kwargs):
            if params.stop_sequences:
                response.generated_text = self._enforce_stop_tokens(response.generated_text, params.stop_sequences)
            logger.info("Chunk received: {}".format(response.generated_text))
            yield GenerationChunk(
                text=response.generated_text or "",
                generation_info=self._create_full_generation_info(response.dict()),
            )
            if run_manager:
                run_manager.on_llm_new_token(token=response.generated_text, response=response)

    def _enforce_stop_tokens(self, text: Optional[str], stop: List[str]):
        """Cut off the text as soon as any stop words occur."""
        if not stop:
            return text or ""

        escaped_stop_sequences = [re.escape(s) for s in stop]
        return re.split("|".join(escaped_stop_sequences), text or "", 1)[0]
