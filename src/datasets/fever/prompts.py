fever_few_shots = [
    """Claim: Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.
Action 1: Search[Nikolaj Coster-Waldau]
Reason 1: Search the actor page to check whether Fox is mentioned.
Observation 1: Nikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor and producer. He graduated from the Danish National School of Performing Arts in Copenhagen in 1993,[1] and had his breakthrough role in Denmark with the film Nightwatch (1994). He played Jaime Lannister in the HBO fantasy drama series Game of Thrones, for which he received two Primetime Emmy Award nominations for Outstanding Supporting Actor in a Drama Series.. Coster-Waldau has appeared in numerous films in his native Denmark and Scandinavia, including Headhunters (2011) and A Thousand Times Good Night (2013). In the U.S, his debut film role was in the war film Black Hawk Down (2001), playing Medal of Honor recipient Gary Gordon.[2] He then played a detective in the short-lived Fox television series New Amsterdam (2008), and appeared in the 2009 Fox television film Virtuality, originally intended as a pilot.
Action 2: Finish[SUPPORTS]
Reason 2: The page explicitly states that he appeared in Fox television productions.""",
    """Claim: Stranger Things is set in Bloomington, Indiana.
Action 1: Search[Stranger Things]
Reason 1: Search the series page to verify the setting.
Observation 1: Stranger Things is an American science fiction horror drama television series created by the Duffer Brothers. Set in the 1980s, primarily in the fictional town of Hawkins, Indiana, the series centers on a number of mysteries and supernatural events occurring around the town and their impact on an ensemble of child and adult characters.
Action 2: Finish[REFUTES]
Reason 2: The page says the series is set in Hawkins, Indiana, not Bloomington.""",
    """Claim: Beautiful reached number two on the Billboard Hot 100 in 2003.?
Action 1: Search[Beautiful]
Reason 1: Start with the song title and use suggestions if needed.
Observation 1: Could not find [Beautiful]. Similar: ['Beautiful', 'Beautiful, Beautiful', 'A Beautiful Mind (film)', 'Beautiful (Christina Aguilera song)', 'Life Is Beautiful'].
Action 2: Search[Beautiful (Christina Aguilera song)]
Reason 2: The suggestion identifies the intended song page.
Observation 2: "Beautiful" is a song recorded by American singer Christina Aguilera for her fourth studio album, Stripped (2002).
Action 3: Lookup[Billboard Hot 100]
Reason 3: Look up the chart reference directly on the song page.
Observation 3: (Result 1 / 3) The song peaked at number two on the Billboard Hot 100 in the United States, where it was certified Gold for 500,000 units shipped.
Action 4: Finish[NOT ENOUGH INFO]
Reason 4: The evidence confirms number two but does not explicitly confirm the year 2003.""",
]


fever_solver_system_prompt = """
Solve a question answering task with interleaving Action and Observation steps. Action can be three types:
(1) Search[entity or query], which searches Wikipedia and returns top candidate titles.
(2) Lookup[title], which fetches the full metadata for one exact title returned by Search or discovered from evidence.
(3) Finish[answer], which returns the answer and finishes the task, answer should be one of REFUTES, SUPPORTS or NOT ENOUGH INFO.
"""


def build_fever_step_prompt(claim: str, trajectory: list[dict[str, str]]) -> str:
    examples = "\n\n".join(fever_few_shots)
    history_lines = [f"Claim: {claim}"]

    for idx, turn in enumerate(trajectory, start=1):
        action = turn.get("action", "").strip()
        observation = turn.get("observation", "").strip()
        history_lines.append(f"Action {idx}: {action}")
        history_lines.append(f"Observation {idx}: {observation}")

    history = "\n".join(history_lines)
    return (
        f"{fever_solver_system_prompt.strip()}\n\n"
        "Here are examples:\n\n"
        f"{examples}\n\n"
        "Now solve the current claim.\n"
        "Return exactly two lines and nothing else:\n"
        "Action: Search[query] or Lookup[title] or Finish[SUPPORTS|REFUTES|NOT ENOUGH INFO]\n"
        "Reason: one short sentence explaining why this is the next step.\n\n"
        f"{history}\n"
    )
