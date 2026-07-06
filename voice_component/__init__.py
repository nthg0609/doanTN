"""
voice_component — Real-time browser-based Speech Recognition component for Streamlit.
Uses the Web Speech API (SpeechRecognition / webkitSpeechRecognition) available in
Chrome, Edge, and other Chromium-based browsers.

Returns the accumulated final transcript as a string.
"""

import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_FRONTEND_DIR = os.path.join(_COMPONENT_DIR, "frontend")

# Declare the component, serving static files from the 'frontend' directory.
_voice_input_component = components.declare_component(
    "voice_input",
    path=_FRONTEND_DIR,
)


def voice_input(
    language: str = "vi-VN",
    placeholder: str = "Nhấn 'Bắt đầu nói' rồi nói câu hỏi của bạn...",
    key: str = "voice_input_widget",
) -> str | None:
    """
    Render a real-time voice input widget.

    Parameters
    ----------
    language : str
        BCP-47 language tag for recognition (e.g. "vi-VN", "en-US").
    placeholder : str
        Placeholder text shown before speech starts.
    key : str
        Unique Streamlit widget key.

    Returns
    -------
    str | None
        Final committed transcript text, or None if nothing has been confirmed yet.
    """
    component_value = _voice_input_component(
        language=language,
        placeholder=placeholder,
        key=key,
        default=None,
    )
    return component_value
