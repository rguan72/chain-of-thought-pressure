"""Prompt templates and utilities."""

from typing import Dict, Any

# Suffix to append to system messages to enforce CoT format
COT_SYSTEM_SUFFIX = """

You must think step-by-step inside <think> tags and provide your final answer inside <final> tags.
Do not include reasoning in <final>. Format your response exactly as:
<think>
[Your step-by-step reasoning here]
</think>
<final>
[Your concise final answer here]
</final>"""


def build_prompt(example: Dict[str, Any], include_cot_instruction: bool = True) -> str:
    """Build a full prompt from a DolusChat example.

    Args:
        example: DolusChat example with system_message, context, user_query fields.
        include_cot_instruction: Whether to append CoT format instruction.

    Returns:
        Formatted prompt string.
    """
    # Extract fields
    system_message = example.get("system_message", "")
    context = example.get("context", {})
    user_query = example.get("user_query", {})

    # Build context string
    context_parts = []
    if context.get("setting"):
        context_parts.append(f"Setting: {context['setting']}")
    if context.get("sensitive_info"):
        context_parts.append(f"Important information: {context['sensitive_info']}")
    if context.get("non_sensitive_info"):
        non_sensitive = context["non_sensitive_info"]
        if isinstance(non_sensitive, list):
            context_parts.append(f"Additional context: {'; '.join(non_sensitive)}")
        else:
            context_parts.append(f"Additional context: {non_sensitive}")

    context_str = "\n".join(context_parts) if context_parts else ""

    # Build user query string
    user_content = user_query.get("content", "")
    speaker_type = user_query.get("speaker_type", "")

    # Compose system message
    full_system = system_message
    if context_str:
        full_system = f"{full_system}\n\n{context_str}"

    if include_cot_instruction:
        full_system = f"{full_system}{COT_SYSTEM_SUFFIX}"

    # Format as chat-style prompt
    prompt = f"""<|im_start|>system
{full_system}
<|im_end|>
<|im_start|>user
{user_content}
<|im_end|>
<|im_start|>assistant
"""

    return prompt


def build_teacher_cot_prompt(example: Dict[str, Any], final_answer: str) -> str:
    """Build a prompt for generating teacher CoT given a known final answer.

    This is used to create CoT training data for the detector.

    Args:
        example: DolusChat example.
        final_answer: The known final answer (truthful or deceptive).

    Returns:
        Prompt that instructs the model to generate CoT leading to the final answer.
    """
    base_prompt = build_prompt(example, include_cot_instruction=False)

    # Remove the assistant turn start and add instruction
    teacher_instruction = f"""Given the conversation above and the following final answer, generate a plausible chain-of-thought reasoning that would lead to this answer. Output only the reasoning inside <think> tags.

Final answer to justify: {final_answer}

<think>"""

    # Build the teacher prompt
    prompt = f"""{base_prompt.rstrip()}
{teacher_instruction}"""

    return prompt


def format_response_with_cot(cot_text: str, final_text: str) -> str:
    """Format a response with CoT in the expected format.

    Args:
        cot_text: The chain-of-thought reasoning.
        final_text: The final answer.

    Returns:
        Formatted response string.
    """
    return f"""<think>
{cot_text}
</think>
<final>
{final_text}
</final>"""
