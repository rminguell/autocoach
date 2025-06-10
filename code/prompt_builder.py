from typing import Union, List, Optional, Dict, Any


def lowercase_first_char(text: str) -> str:
    return text[0].lower() + text[1:] if text else text


def format_prompt_section(lead_in: str, value: Union[str, List[str]]) -> str:
    if isinstance(value, list):
        formatted_value = "\n".join(f"- {item}" for item in value)
    else:
        formatted_value = value
    return f"{lead_in}\n{formatted_value}"


def build_prompt_from_config(
    config: Dict[str, Any],
    input_data: str = "",
    app_config: Optional[Dict[str, Any]] = None,
) -> str:
    prompt_parts = []

    if role := config.get("role"):
        prompt_parts.append(f"You are {lowercase_first_char(role.strip())}.")

    instruction = config.get("instruction")
    if not instruction:
        raise ValueError("Missing required field: 'instruction'")
    prompt_parts.append(format_prompt_section("Your task is as follows:", instruction))

    if context := config.get("context"):
        prompt_parts.append(f"Here’s some background that may help you:\n{context}")

    if constraints := config.get("output_constraints"):
        prompt_parts.append(
            format_prompt_section(
                "Ensure your response follows these rules:", constraints
            )
        )

    if tone := config.get("style_or_tone"):
        prompt_parts.append(
            format_prompt_section(
                "Follow these style and tone guidelines in your response:", tone
            )
        )

    if format_ := config.get("output_format"):
        prompt_parts.append(
            format_prompt_section("Structure your response as follows:", format_)
        )

    if examples := config.get("examples"):
        prompt_parts.append("Here are some examples to guide your response:")
        if isinstance(examples, list):
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"Example {i}:\n{example}")
        else:
            prompt_parts.append(str(examples))

    if goal := config.get("goal"):
        prompt_parts.append(f"Your goal is to achieve the following outcome:\n{goal}")

    if input_data:
        prompt_parts.append(
            "Here is the content you need to work with:\n"
            "<<<BEGIN CONTENT>>>\n"
            "```\n" + input_data.strip() + "\n```\n<<<END CONTENT>>>"
        )

    reasoning_strategy = config.get("reasoning_strategy")
    if reasoning_strategy and reasoning_strategy != "None" and app_config:
        strategies = app_config.get("reasoning_strategies", {})
        if strategy_text := strategies.get(reasoning_strategy):
            prompt_parts.append(strategy_text.strip())

    prompt_parts.append("Now perform the task as instructed above.")
    return "\n\n".join(prompt_parts)


def build_system_prompt_from_config(
    config: Dict[str, Any],
    publication_content: str = "",
) -> str:
    prompt_parts = []

    role = config.get("role")
    if not role:
        raise ValueError("Missing required field: 'role'")
    prompt_parts.append(f"You are {lowercase_first_char(role.strip())}.")

    if constraints := config.get("output_constraints"):
        prompt_parts.append(format_prompt_section("Follow these important guidelines:", constraints))

    if tone := config.get("style_or_tone"):
        prompt_parts.append(format_prompt_section("Communication style:", tone))

    if format_ := config.get("output_format"):
        prompt_parts.append(format_prompt_section("Response formatting:", format_))

    if goal := config.get("goal"):
        prompt_parts.append(f"Your primary objective: {goal}")

    if publication_content:
        prompt_parts.append(
            "Base your responses on this publication content:\n\n"
            "=== PUBLICATION CONTENT ===\n"
            f"{publication_content.strip()}\n"
            "=== END PUBLICATION CONTENT ==="
        )

    return "\n\n".join(prompt_parts)