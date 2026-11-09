import argparse
import asyncio
import json

import llm_openrouter_decisions as plugin


def main(endpoint):
    if not endpoint.startswith("http://127.0.0.1:"):
        raise ValueError("Use a local contract endpoint")
    plugin.API_URL = endpoint + "/api/alpha/decisions"
    questions = {
        "category": {"type": "choice", "instructions": "Classify the request",
                     "criteria": {"account": "Login problems", "billing": "Refund requests"}},
        "defect": {"type": "noul", "instructions": "Is the software broken?"},
        "urgency": {"type": "score", "instructions": "How urgent is it?",
                    "criteria": ["Can wait", "Soon", "Blocked"]},
    }
    model = plugin.Decision("venn-1")
    response = model.prompt("The login page is broken", key="local-contract-only", questions=questions)
    answers = json.loads(response.text())
    assert set(answers) == set(questions)
    assert answers["category"]["choice"] in ("account", "billing")
    assert 0 <= answers["defect"]["noul"] <= 1
    assert 0 <= answers["urgency"]["score"] <= 2
    assert response.response_json["model"] == "venn-1"

    async def check_async():
        response = plugin.AsyncDecision("venn-1").prompt(
            "The login page is broken", key="local-contract-only", questions=questions)
        assert json.loads(await response.text()) == answers
    asyncio.run(check_async())
    print(json.dumps({"client": "llm-openrouter-decisions", "passed": ["sync", "async",
                      "choice", "noul", "score"], "source_revision":
                      "d22bc2b0ef33802674ca6d3ce01da478946dc34f"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint")
    main(parser.parse_args().endpoint)
