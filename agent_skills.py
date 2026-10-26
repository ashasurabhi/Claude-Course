import os
import anthropic
from anthropic.lib import files_from_dir
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from the .env file

client = anthropic.Anthropic()
MODEL = "claude-haiku-5-5"
SKILL_DIR = "status-report-skill"
ID_FILE = "skill_id.txt"

ACTIVITY_LOG = """
09:10 Priya: finished the Q3 forecast model
10:30 Tom: waiting on finance for the March invoice data
11:00 Lena: started the onboarding checklist, owner Lena
14:20 Tom: fixed the duplicate customer keys in the staging load
16:00 Priya: planning the Thursday model review
"""


def get_or_create_skill() -> str:
    """Upload the skill folder the first time, then reuse the saved ID."""
    if os.path.exists(ID_FILE):
        with open(ID_FILE) as f:
            return f.read().strip()

    skill = client.skills.create(
        display_name="Status Report Generator",
        files=files_from_dir(SKILL_DIR),
    )

    with open(ID_FILE, "w") as f:
        f.write(skill.id)
    print(f"Created skill {skill.id} and saved it to {ID_FILE}")
    return skill.id


def write_report(activity_log: str, skill_id: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        container={
            "skills": [
                {"type": "custom", "skill_id": skill_id, "version": "latest"}
            ]
        },
        tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
        messages=[
            {
                "role": "user",
                "content": f"Generate the daily status report from this activity log.Return only the report text, with no commentary\n\n{activity_log}",
            }
        ],
    )

    # The skill may run code, so the reply can include code blocks.
    # Keep only the text, which holds the report.
    report_text = "".join(block.text for block in response.content if block.type == "text")

    with open("daily_status_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


if __name__ == "__main__":
    skill_id = get_or_create_skill()
    print(write_report(ACTIVITY_LOG, skill_id))