import logging
import random
from types import SimpleNamespace

from .llm.llm import LLMs
from .logger import get_logger

logger = get_logger(__name__, level=logging.INFO)


class TurnManager:
    def __init__(self, cfg: SimpleNamespace, llms: LLMs):
        self.cfg = cfg
        self.mode = cfg.chat.turn_control_mode
        self.llms = llms

        # Consistent order of speakers from config
        self.all_speakers = [self.llms.llmcfg.user_name] + [
            ai.name for ai in self.cfg.chat.ai
        ]

        # Current index for round_robin
        self.current_speaker_index = 0
        self.reset()

    def reset(self):
        """Resets the turn manager to its initial state based on config."""
        try:
            initial_turn = self.llms.llmcfg.format(self.cfg.chat.initial_turn)
            self.current_speaker_index = self.all_speakers.index(initial_turn)
        except (ValueError, AttributeError):
            logger.warning(
                "Initial turn speaker not found in all_speakers list. Defaulting to index 0."
            )
            self.current_speaker_index = 0
        logger.debug(
            f"TurnManager reset. Mode: {self.mode}. Initial speaker: {self.all_speakers[self.current_speaker_index]}"
        )

    def _get_next_speaker_random(self, last_speaker: str) -> str:
        """Selects the next speaker randomly, excluding the last speaker."""
        possible_choices = [s for s in self.all_speakers if s != last_speaker]
        if not possible_choices:
            # This happens if there's only one speaker in total.
            return last_speaker
        return random.choice(possible_choices)

    def _get_next_speaker_round_robin(self, last_speaker: str) -> str:
        """Selects the next speaker in a round-robin fashion."""
        try:
            last_speaker_index = self.all_speakers.index(last_speaker)
        except ValueError:
            # Fallback if last_speaker is not in the list for some reason
            logger.warning(
                f"Last speaker '{last_speaker}' not found. Using internal index."
            )
            last_speaker_index = self.current_speaker_index

        next_index = (last_speaker_index + 1) % len(self.all_speakers)
        self.current_speaker_index = next_index
        return self.all_speakers[next_index]

    def get_next_speaker(self, history: list[dict], last_speaker: str) -> str:
        """Determines the next speaker based on the configured mode, always excluding the last speaker."""
        except_names = [last_speaker]

        try:
            if self.mode == "llm":
                return self.llms.get_next_speaker(history, except_names)

            if self.mode == "random":
                return self._get_next_speaker_random(last_speaker)

            if self.mode == "round_robin":
                return self._get_next_speaker_round_robin(last_speaker)

            # Fallback for unknown mode
            logger.warning(
                f"Unknown turn_control_mode: {self.mode}. Falling back to 'random' mode."
            )
            return self._get_next_speaker_random(last_speaker)

        except Exception as e:
            logger.warning(
                f"Failed to get next speaker with mode '{self.mode}' (Error: {e}). "
                "Falling back to random selection."
            )
            return self._get_next_speaker_random(last_speaker)
