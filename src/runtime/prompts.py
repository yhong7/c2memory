SUPERVISOR_PROMPT = (
    "You are the supervisor of a graph-structured multi-agent system. "
    "Your job is to coordinate, not to solve the task directly. "
    "You should identify the current subgoal, highlight what evidence or entity should be checked next, "
    "and give a short brief that helps downstream agents act. "
    "Do not output environment actions such as Search[...], Lookup[...], or Finish[...]. "
    "Your output is invalid if it contains an action."
)

SUB_AGENT_PROMPT = (
    "You are a reusable sub-agent. "
    "Follow the provided supervisor brief and solve one slice of the task."
)

FINAL_PROMPT = (
    "You are the final agent. "
    "Combine the sub-agent outputs into a concise final answer. "
    "Output ONLY the answer with NO extra text, explanation, or formatting."
)

FEVER_FINAL_PROMPT = (
    "You are the final agent for a FEVER-style fact verification task. "
    "Return exactly two lines and nothing else:\n"
    "Action: Search[query] or Lookup[title] or Finish[SUPPORTS|REFUTES|NOT ENOUGH INFO]\n"
    "Reason: one short sentence explaining why this is the next step."
)

HOTPOTQA_FINAL_PROMPT = (
    "You are the final commit agent for a HotpotQA-style multi-hop retrieval task. "
    "Your job is to validate the upstream decision and convert it into the ONE concrete environment action "
    "that will actually be executed next. "
    "Return exactly two lines and nothing else:\n"
    "Action: Search[query] or Lookup[title] or Finish[answer]\n"
    "Reason: one short sentence explaining why this is the next step.\n"
    "Do not preserve an upstream Finish merely because its syntax is valid. Finish only when committed "
    "observations explicitly support the question's exact outermost answer slot. For yes/no questions, "
    "use exactly Finish[yes] or Finish[no], and verify every compared entity first. For other questions, "
    "put only the shortest answer span inside Finish; never include clues, reference entities, or explanations."
)

DYLAN_VALID_ROLES: list[str] = ["solver", "ground_truth", "decision"]

DYLAN_SOLVER_PROMPT = (
    "You are a smart agent designed to solve problems. "
    "You must strictly follow the output format used by other agents "
    "and must not deviate from or alter the format specifications."
)

DYLAN_GROUND_TRUTH_PROMPT = (
    "You are an intelligent agent equipped with common sense and real-world knowledge, "
    "specifically designed to solve problems based on ground truth. "
    "When reasoning and answering questions, you must rely on realistic assumptions "
    "and provide responses that align with common sense and real-world logic. "
    "You must strictly follow the output format used by other agents "
    "and must not deviate from or alter the format specifications."
)

DYLAN_DECISION_PROMPT = (
    "You are an intelligent agent responsible for making the final decision. "
    "You will be given responses from several other agents regarding the current task. "
    "Your role is to carefully consider all their answers and provide your own final answer "
    "based on their input.\n\n"
    "NOTES:\n"
    "- If you believe all other agents' answers are incorrect, you may provide your own independent answer.\n"
    "- You must strictly adhere to the output format used by the other agents. "
    "Do not deviate from or modify the formatting rules."
)

DYLAN_RANKER_PROMPT = (
    "You are an evaluation agent responsible for ranking the correctness of outputs "
    "provided by multiple agents. Given several agents' responses to a task, you must "
    "compare them and rank their correctness from most to least accurate.\n\n"
    "Output Format:\n"
    "- Your output must be in the exact format: 1 > 3 > 2 "
    "(representing the agent numbers in descending order of correctness).\n"
    "- Do NOT include any extra text, explanations, or symbols outside of this format.\n"
    "- The ranking should reflect only the correctness of the responses."
)

DYLAN_FEVER_SOLVER_PROMPT = (
    "You are a fact-verification agent. Your job is to EXACTLY match a claim against Wikipedia evidence. "
    "For each claim: use Search[query] to retrieve candidate Wikipedia titles, then Lookup[title] "
    "to inspect the exact evidence page. "
    "CRITICAL RULES:\n"
    "1. Finding that entity X has attribute A does NOT refute a claim that X has attribute B. "
    "Only answer REFUTES if Wikipedia EXPLICITLY states the claim is false.\n"
    "2. If Wikipedia says a company makes cars, that does NOT refute 'the company makes motorcycles' "
    "— it only refutes if Wikipedia explicitly says 'the company does NOT make motorcycles'.\n"
    "3. Answer NOT ENOUGH INFO when Wikipedia does not explicitly address every part of the claim. "
    "When in doubt, answer NOT ENOUGH INFO.\n"
    "4. Do NOT infer, deduce, or connect separate facts — only explicit statements count as evidence.\n"
    "Follow the output format strictly."
)

DYLAN_HOTPOTQA_SOLVER_PROMPT = (
    "You are the exploratory solver in a HotpotQA DyLAN layer. "
    "Your responsibility is to propose the most promising NEXT move, even if the evidence is not "
    "yet sufficient to commit a final action. "
    "You must output exactly one concrete next action plus one short reason. "
    "Prefer Search when you need candidate titles for a missing entity or relation. "
    "Prefer Lookup when you already know which title should be inspected next. "
    "Finish only when the answer is already supported by retrieved evidence. "
    "Return exactly two lines and nothing else:\n"
    "Action: Search[query] or Lookup[title] or Finish[answer]\n"
    "Reason: one short sentence explaining why this is the next step."
)

DYLAN_HOTPOTQA_GROUND_TRUTH_PROMPT = (
    "You are the evidence-checking agent in a HotpotQA DyLAN layer. "
    "Your responsibility is to stay close to explicit retrieved evidence and to identify what exact "
    "fact is still missing. "
    "You must output exactly one concrete next action plus one short reason. "
    "Prefer Lookup when a specific retrieved title is the most likely source of the missing evidence. "
    "Prefer Search only when the relevant title has not been surfaced yet. "
    "Do not guess an answer from weak evidence. "
    "Return exactly two lines and nothing else:\n"
    "Action: Search[query] or Lookup[title] or Finish[answer]\n"
    "Reason: one short sentence explaining why this is the next step."
)

DYLAN_HOTPOTQA_DECISION_PROMPT = (
    "You are the coordination agent inside a HotpotQA DyLAN layer. "
    "Your responsibility is to look at the current question state and prefer the most useful NEXT step "
    "for the whole layer. "
    "You must output exactly one concrete next action plus one short reason. "
    "Choose the action that best advances the whole layer. "
    "Return exactly two lines and nothing else:\n"
    "Action: Search[query] or Lookup[title] or Finish[answer]\n"
    "Reason: one short sentence explaining why this is the next step."
)

DYLAN_HOTPOTQA_AGGREGATOR_PROMPT = (
    "You are the DyLAN decision aggregator for a HotpotQA retrieval step. "
    "You will receive several candidate outputs from the current layer. "
    "Your job is to synthesize them into the ONE best next environment action. "
    "Choose Finish only when the answer is already supported by retrieved evidence. "
    "If any agent already proposes a well-supported Finish[answer], prefer that Finish over repeating Search/Lookup. "
    "When you choose Finish, use the shortest exact answer span supported by the evidence.\n"
    "Return exactly two lines and nothing else:\n"
    "Action: Search[query] or Lookup[title] or Finish[answer]\n"
    "Reason: one short sentence explaining why this is the best next step."
)

DYLAN_ROLE_MAP: dict[str, str] = {
    "solver": DYLAN_SOLVER_PROMPT,
    "ground_truth": DYLAN_GROUND_TRUTH_PROMPT,
    "decision": DYLAN_DECISION_PROMPT,
    "ranker": DYLAN_RANKER_PROMPT,
}

LLM_DEBATE_VALID_ROLES: list[str] = ["debater", "solver", "critic", "fact_checker", "judge"]

LLM_DEBATE_ROLE_PROMPTS: dict[str, str] = {
    "debater": (
        "You are a careful agent in a multi-agent debate system. "
        "First propose the best next environment action independently. In later rounds, "
        "read other agents' proposals, keep your own judgment, and revise only when their "
        "evidence is stronger. Always output one concrete next action."
    ),
    "solver": (
        "You are a precise solver in a debate system. Focus on the action that most directly "
        "advances the current task state. Verify evidence before proposing Finish."
    ),
    "critic": (
        "You are a critical debater. Find unsupported Finish actions, repeated searches, "
        "wrong lookup targets, and hidden assumptions. Still output your own best next action."
    ),
    "fact_checker": (
        "You are a fact-checking debater. Stay grounded in the latest observations and prefer "
        "actions that obtain or use explicit evidence. Do not guess unsupported answers."
    ),
    "judge": (
        "You are the final decision agent in a multi-agent debate system. Compare the agents' "
        "latest proposals, identify agreement and conflict, and choose one concrete next action."
    ),
}


def build_llm_debate_initial_prompt(
    *,
    task: str,
    supervisor_brief: str,
    memory_context: list[str],
    behavior_memory_context: list[str] | None = None,
) -> str:
    memory_block = "\n".join(memory_context) if memory_context else "(empty memory)"
    return (
        "Solve the current MAS step independently by proposing the best next environment action.\n\n"
        f"# Task\n{task}\n\n"
        f"# Supervisor brief\n{supervisor_brief}\n\n"
        f"{_render_behavior_memory_block(behavior_memory_context)}"
        f"# Memory context\n{memory_block}\n\n"
        "Return exactly two lines and nothing else:\n"
        "Action: <one concrete environment action>\n"
        "Reason: one short sentence explaining why this is the next step.\n"
        "When using Finish[answer], put only the shortest final answer span inside brackets. "
        "For yes/no questions, use exactly Finish[yes] or Finish[no]."
    )


def build_llm_debate_round_prompt(
    *,
    task: str,
    supervisor_brief: str,
    self_previous_answer: str,
    other_answers: list[tuple[str, str, str]],
    round_index: int,
    memory_context: list[str],
    behavior_memory_context: list[str] | None = None,
) -> str:
    memory_block = "\n".join(memory_context) if memory_context else "(empty memory)"
    other_block = (
        "\n\n".join(
            f"### {name} ({role})\n{answer}"
            for name, role, answer in other_answers
        )
        if other_answers
        else "(no other answers)"
    )
    return (
        f"We are in debate round {round_index}. Re-evaluate the best next environment action.\n\n"
        f"# Task\n{task}\n\n"
        f"# Supervisor brief\n{supervisor_brief}\n\n"
        f"{_render_behavior_memory_block(behavior_memory_context)}"
        f"# Memory context\n{memory_block}\n\n"
        f"# Your previous answer\n{self_previous_answer or '(none)'}\n\n"
        f"# Other agents' latest answers\n{other_block}\n\n"
        "Compare the proposals, keep or revise your action, and return exactly two lines:\n"
        "Action: <one concrete environment action>\n"
        "Reason: one short sentence explaining why this is the next step.\n"
        "When using Finish[answer], put only the shortest final answer span inside brackets. "
        "For yes/no questions, use exactly Finish[yes] or Finish[no]."
    )


def build_llm_debate_judge_prompt(
    *,
    task: str,
    supervisor_brief: str,
    final_answers: list[tuple[str, str, str]],
    behavior_memory_context: list[str] | None = None,
) -> str:
    answers_block = "\n\n".join(
        f"### Agent {idx + 1}: {name} ({role})\n{answer}"
        for idx, (name, role, answer) in enumerate(final_answers)
    )
    return (
        "Make the final debate decision for this MAS step.\n\n"
        f"# Task\n{task}\n\n"
        f"# Supervisor brief\n{supervisor_brief}\n\n"
        f"{_render_behavior_memory_block(behavior_memory_context)}"
        f"# Agents' final answers\n{answers_block}\n\n"
        "Choose the single best next environment action. If all agents are wrong, provide a corrected action.\n"
        "Return exactly two lines and nothing else:\n"
        "Action: <one concrete environment action>\n"
        "Reason: one short sentence explaining why this is the next step.\n"
        "When using Finish[answer], put only the shortest final answer span inside brackets. "
        "For yes/no questions, use exactly Finish[yes] or Finish[no]."
    )


def build_subagent_core_prompt(task: str, supervisor_brief: str, memory_context: list[str]) -> str:
    memory_block = "\n".join(memory_context) if memory_context else "(empty memory)"
    return (
        "Internal sub-agent pipeline:\n"
        "1. Input accepted\n"
        "2. Memory retrieved\n"
        "3. Core execution\n"
        "4. Memory updated\n\n"
        f"Task:\n{task}\n\n"
        f"Supervisor brief:\n{supervisor_brief}\n\n"
        f"Memory context:\n{memory_block}\n\n"
        "Return only your sub-agent result."
    )


def _render_behavior_memory_block(behavior_memory_context: list[str] | None) -> str:
    if not behavior_memory_context:
        return ""
    joined = "\n\n".join(behavior_memory_context)
    return f"# C2Mem behavior memory\n{joined}\n\n"


HOTPOTQA_ACTION_SCHEMA_NOTE = (
    "Action schema templates: Search[<actual query>], Lookup[<actual title>], "
    "or Finish[<actual answer>]. Angle-bracket names describe required values; "
    "do not copy placeholder words such as query, title, or answer as the bracket content. "
    "Finish must contain only the shortest exact answer span, without an explanatory sentence. "
    "For yes/no questions, use exactly Finish[yes] or Finish[no]. Answer only the requested "
    "outermost slot; do not include entities supplied by the question merely as clues or reference points. "
    "For example, in 'an album by a group that was formed by who?', the group is an intermediate entity "
    "and the requested slot is the founder. In a comparison, verify both entities before Finish; evidence "
    "about only one entity is insufficient.\n"
)


def build_dylan_layer_prompt(task: str, supervisor_brief: str, upstream_context: str, memory_context: list[str], behavior_memory_context: list[str] | None = None) -> str:
    memory_block = "\n".join(memory_context) if memory_context else "(empty memory)"
    return (
        f"# Task\n{task}\n\n"
        f"# Supervisor brief\n{supervisor_brief}\n\n"
        f"{_render_behavior_memory_block(behavior_memory_context)}"
        f"{upstream_context}# Memory context\n{memory_block}\n"
        "----------------------------------------\n"
        "Provide your answer for this task.\n"
        "Your output:"
    )


def build_dylan_ranker_prompt(agents_responses: str) -> str:
    return (
        "## Below are the answers provided by other agents\n"
        f"{agents_responses}\n\nYour output:"
    )


def build_dylan_decision_prompt(task: str, supervisor_brief: str, agents_responses: str, behavior_memory_context: list[str] | None = None) -> str:
    return (
        f"# Task\n{task}\n\n"
        f"# Supervisor brief\n{supervisor_brief}\n\n"
        f"{_render_behavior_memory_block(behavior_memory_context)}"
        "## Below are the answers provided by other agents\n"
        f"{agents_responses}\n\nYour output:"
    )


def build_dylan_final_fever_prompt(task: str, supervisor_brief: str, final_output: str, behavior_memory_context: list[str] | None = None) -> str:
    return (
        f"Claim:\n{task}\n\n"
        f"Supervisor brief:\n{supervisor_brief}\n\n"
        f"{_render_behavior_memory_block(behavior_memory_context)}"
        f"DyLAN decision:\n{final_output}\n\n"
        "Return exactly two lines and nothing else:\n"
        "Action: Search[query] or Lookup[title] or Finish[SUPPORTS|REFUTES|NOT ENOUGH INFO]\n"
        "Reason: one short sentence explaining why this is the next step."
    )


def build_dylan_final_prompt(task: str, supervisor_brief: str, final_output: str, behavior_memory_context: list[str] | None = None) -> str:
    return (
        f"Task:\n{task}\n\n"
        f"Supervisor brief:\n{supervisor_brief}\n\n"
        f"{_render_behavior_memory_block(behavior_memory_context)}"
        f"DyLAN decision:\n{final_output}\n\n"
        "Output ONLY the final answer with NO extra text or formatting."
    )


def build_supervisor_prompt(task: str, behavior_memory_context: list[str] | None = None) -> str:
    return (
        "Write a supervisor brief for the downstream agents.\n"
        "Return EXACTLY these three lines and nothing else:\n"
        "Current focus: ...\n"
        "Why this matters: ...\n"
        "Next evidence target: ...\n\n"
        "Do not output Search[...], Lookup[...], Finish[...], or a final answer.\n"
        "If you output an action, your answer is invalid.\n\n"
        "Example brief:\n"
        "Current focus: verify the first entity's nationality.\n"
        "Why this matters: the question asks whether two people share the same nationality.\n"
        "Next evidence target: find an explicit statement about the first person's nationality.\n\n"
        f"{_render_behavior_memory_block(behavior_memory_context)}"
        f"Task:\n{task}\n"
    )


def build_final_fever_prompt(task: str, supervisor_brief: str, rendered_outputs: str, behavior_memory_context: list[str] | None = None) -> str:
    return (
        f"Claim:\n{task}\n\n"
        f"Supervisor brief:\n{supervisor_brief}\n\n"
        f"{_render_behavior_memory_block(behavior_memory_context)}"
        f"Sub-agent outputs:\n{rendered_outputs}\n\n"
        "Return exactly two lines and nothing else:\n"
        "Action: Search[query] or Lookup[title] or Finish[SUPPORTS|REFUTES|NOT ENOUGH INFO]\n"
        "Reason: one short sentence explaining why this is the next step."
    )


def build_final_hotpotqa_prompt(task: str, supervisor_brief: str, rendered_outputs: str, behavior_memory_context: list[str] | None = None) -> str:
    return (
        f"Question:\n{task}\n\n"
        f"Supervisor brief:\n{supervisor_brief}\n\n"
        f"{_render_behavior_memory_block(behavior_memory_context)}"
        f"Sub-agent outputs:\n{rendered_outputs}\n\n"
        "Commit the next environment action now. Use the sub-agent outputs as evidence, "
        "and apply C2Mem behavior memory as behavior constraints when it is present.\n"
        f"{HOTPOTQA_ACTION_SCHEMA_NOTE}"
        "Return exactly two lines and nothing else:\n"
        "Action: Search[...] or Lookup[...] or Finish[...]\n"
        "Reason: one short sentence explaining why this is the next step."
    )


def build_dylan_final_hotpotqa_prompt(task: str, supervisor_brief: str, final_output: str, behavior_memory_context: list[str] | None = None) -> str:
    return (
        f"{_render_behavior_memory_block(behavior_memory_context)}"
        f"Question:\n{task}\n\n"
        f"Supervisor brief:\n{supervisor_brief}\n\n"
        "Decision candidate from the previous attempt:\n"
        f"{final_output}\n\n"
        "Commit the next environment action now. Treat C2Mem behavior memory as the "
        "highest-priority behavior constraint for this recommit. Preserve the candidate's "
        "evidence basis when it is still valid, but do not preserve a flawed submitted "
        "action exactly when the memory identifies a formatting, stopping, or answer-span "
        "problem.\n"
        f"{HOTPOTQA_ACTION_SCHEMA_NOTE}"
        "Return exactly two lines and nothing else:\n"
        "Action: Search[...] or Lookup[...] or Finish[...]\n"
        "Reason: one short sentence explaining why this is the next step."
    )


def build_final_prompt(task: str, supervisor_brief: str, rendered_outputs: str, behavior_memory_context: list[str] | None = None) -> str:
    return (
        f"Task:\n{task}\n\n"
        f"Supervisor brief:\n{supervisor_brief}\n\n"
        f"{_render_behavior_memory_block(behavior_memory_context)}"
        f"Sub-agent outputs:\n{rendered_outputs}\n\n"
        "Return the final answer."
    )
