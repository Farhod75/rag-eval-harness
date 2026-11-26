"""Golden dataset: questions with known answers (ground truth) from data/handbook."""

# (question, fact that must appear in the answer, chunk id that must be cited)
ANSWERABLE = [
    ("How much is reimbursed for meals per day on a business trip?", "$75", "expenses#2"),
    ("Who must sign off on travel expenses above $500?", "VP", "expenses#1"),
    ("How long do I have to return my laptop after leaving the company?", "5 business days", "equipment#1"),
    ("How quickly must a stolen laptop be reported?", "24 hours", "equipment#2"),
    ("What are the core hours for remote employees?", "10:00", "remote_work#2"),
]

# Not covered by any document -> must answer NOT_FOUND
UNANSWERABLE = [
    "What is the parental leave policy?",
    "Does the company match 401(k) contributions?",
    "Are dogs allowed in the office?",
]

ALL_QUESTIONS = [q for q, _, _ in ANSWERABLE] + UNANSWERABLE