import csv
import json
import os
import re
import requests
from dotenv import load_dotenv


# Load the API key from .env
load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "allam-2-7b"

ALLOWED_BRANCHES = ["CSE", "IT", "AIDS", "ECE", "EEE", "MECH", "CIVIL"]

EMPTY_CRITERIA = {
    "branches": [],
    "min_cgpa": None,
    "max_backlogs": None,
    "batch": None,
    "summary": "Criteria could not be extracted"
}


# Read job details from jobs.csv
def load_jobs():
    with open("jobs.csv", "r", newline="", encoding="utf-8-sig") as file:

        # Remove empty lines from the CSV file
        lines = [line for line in file if line.strip()]

        # Read job records
        reader = csv.DictReader(lines)
        jobs = list(reader)

    return jobs


# Read student details from students.csv
def load_students():
    with open("students.csv", "r", newline="", encoding="utf-8-sig") as file:
        lines = [line for line in file if line.strip()]
        reader = csv.DictReader(lines)
        students = list(reader)

    for student in students:
        student["cgpa"] = float(student["cgpa"])
        student["backlogs"] = int(student["backlogs"])
        student["batch"] = int(student["batch"])

    return students


# Check whether the AI returned valid criteria
def is_valid(data):
    required_fields = {
        "branches", "min_cgpa", "max_backlogs", "batch", "summary"
    }

    if not isinstance(data, dict) or set(data.keys()) != required_fields:
        return False

    branches = data["branches"]

    if not isinstance(branches, list):
        return False

    if "ALL" in branches:
        if branches != ["ALL"]:
            return False
    elif any(branch not in ALLOWED_BRANCHES for branch in branches):
        return False

    cgpa = data["min_cgpa"]
    if cgpa is not None:
        if isinstance(cgpa, bool) or not isinstance(cgpa, (int, float)):
            return False
        if not 0 <= cgpa <= 10:
            return False

    backlogs = data["max_backlogs"]
    if backlogs is not None:
        if isinstance(backlogs, bool) or not isinstance(backlogs, int):
            return False
        if backlogs < 0:
            return False

    batch = data["batch"]
    if batch is not None:
        if isinstance(batch, bool) or not isinstance(batch, int):
            return False
        if not 2000 <= batch <= 2099:
            return False

    if not isinstance(data["summary"], str):
        return False

    if len(data["summary"].split()) > 12:
        return False

    return True


# Ask Groq AI to extract criteria from a job post
def read_criteria(job_post):
    try:
        with open("system_prompt.txt", "r", encoding="utf-8") as file:
            system_prompt = file.read()
    except OSError:
        print("Error: system_prompt.txt could not be read.")
        return EMPTY_CRITERIA.copy()

    if not API_KEY:
        print("Error: API key not found in .env.")
        return EMPTY_CRITERIA.copy()

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    for attempt in range(2):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": job_post}
        ]

        if attempt == 1:
            messages.append({
                "role": "user",
                "content": "Your previous response was invalid. Return only valid JSON with exactly the five required fields."
            })

        data = {
            "model": MODEL,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0
        }

        try:
            response = requests.post(
                API_URL,
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()

            answer = response.json()["choices"][0]["message"]["content"]
            criteria = json.loads(answer)

            if is_valid(criteria):
                return criteria

            print("AI returned invalid criteria. Retrying...")

        except (requests.exceptions.RequestException, ValueError, KeyError, TypeError) as error:
            print("AI request or response error:", error)

    print("AI extraction failed. Sending this job for manual review.")
    return EMPTY_CRITERIA.copy()


# Check AI criteria against the original job post
def check_facts(job_post, criteria):
    post = job_post.upper()

    # Keep only branches explicitly mentioned in the job post
    branch_aliases = {
        "CSE": [r"\bCSE\b", r"\bCOMPUTER SCIENCE\b"],
        "IT": [r"\bIT\b", r"\bINFORMATION TECHNOLOGY\b"],
        "AIDS": [r"\bAIDS\b", r"\bAI\s*&\s*DS\b", r"\bAI\s+AND\s+DS\b"],
        "ECE": [r"\bECE\b", r"\bELECTRONICS AND COMMUNICATION\b"],
        "EEE": [r"\bEEE\b", r"\bELECTRICAL AND ELECTRONICS\b"],
        "MECH": [r"\bMECH\b", r"\bMECHANICAL\b"],
        "CIVIL": [r"\bCIVIL\b"]
    }

    verified_branches = []

    if re.search(r"\bALL BRANCHES\b|\bANY BRANCH\b", post):
        verified_branches = ["ALL"]
    elif criteria["branches"] != ["ALL"]:
        for branch in criteria["branches"]:
            patterns = branch_aliases.get(branch, [])
            if any(re.search(pattern, post) for pattern in patterns):
                verified_branches.append(branch)

    criteria["branches"] = verified_branches

    # Verify the CGPA cutoff against the original post
    cgpa_pattern = (
        r"(?:CGPA|GPA|GRADE POINT AVERAGE)"
        r"\s*(?:OF|:|>=|IS|AT LEAST)?\s*(\d+(?:\.\d+)?)"
    )
    cgpa_match = re.search(cgpa_pattern, post)

    reverse_cgpa_pattern = (
        r"(\d+(?:\.\d+)?)\s*(?:MINIMUM\s*)?(?:CGPA|GPA)\b"
    )
    if not cgpa_match:
        cgpa_match = re.search(reverse_cgpa_pattern, post)

    if cgpa_match:
        stated_cgpa = float(cgpa_match.group(1))
        if criteria["min_cgpa"] != stated_cgpa:
            criteria["min_cgpa"] = None
    else:
        criteria["min_cgpa"] = None

    # Verify the batch year
    years = re.findall(
        r"(?:BATCH|CLASS OF|GRADUATING CLASS)\s*(?:OF\s*)?(20\d{2})",
        post
    )

    if criteria["batch"] not in [int(year) for year in years]:
        criteria["batch"] = None

    # Verify backlog criteria
    if criteria["max_backlogs"] == 0:
        if not re.search(
            r"\bNO (?:ACTIVE )?BACKLOGS?\b|\bZERO BACKLOGS?\b",
            post
        ):
            criteria["max_backlogs"] = None

    elif criteria["max_backlogs"] is not None:
        number = str(criteria["max_backlogs"])
        backlog_patterns = [
            rf"\b(?:MAXIMUM|MAX|UP TO|AT MOST)\s+{number}\s+BACKLOGS?\b",
            rf"\b{number}\s+BACKLOGS?\s+(?:ALLOWED|PERMITTED)\b"
        ]

        if not any(re.search(pattern, post) for pattern in backlog_patterns):
            criteria["max_backlogs"] = None

    # Limit summary to 12 words
    criteria["summary"] = " ".join(criteria["summary"].split()[:12])

    return criteria


# Decide whether a job can be automatically processed
def job_status(criteria):
    if not criteria["branches"] or criteria["min_cgpa"] is None:
        return "needs_review"

    return "matched"


# Check whether one student meets one job's criteria
def is_eligible(student, criteria):
    branch_match = (
        criteria["branches"] == ["ALL"]
        or student["branch"] in criteria["branches"]
    )

    cgpa_match = student["cgpa"] >= criteria["min_cgpa"]

    backlog_match = (
        criteria["max_backlogs"] is None
        or student["backlogs"] <= criteria["max_backlogs"]
    )

    batch_match = (
        criteria["batch"] is None
        or student["batch"] == criteria["batch"]
    )

    return branch_match and cgpa_match and backlog_match and batch_match


# Write all four output reports
def write_reports(job_results, students):
    job_summary_rows = []
    eligible_rows = []
    review_rows = []

    for result in job_results:
        job = result["job"]
        criteria = result["criteria"]
        status = result["status"]

        job_summary_rows.append({
            "job_id": job["id"],
            "company": job["company"],
            "branches": ", ".join(criteria["branches"]),
            "min_cgpa": criteria["min_cgpa"],
            "max_backlogs": criteria["max_backlogs"],
            "batch": criteria["batch"],
            "summary": criteria["summary"],
            "status": status
        })

        if status == "needs_review":
            review_rows.append({
                "job_id": job["id"],
                "company": job["company"],
                "post": job["post"],
                "reason": "Branch or CGPA criterion is missing or unclear"
            })
            continue

        for student in students:
            if is_eligible(student, criteria):
                eligible_rows.append({
                    "job_id": job["id"],
                    "company": job["company"],
                    "roll": student["roll"],
                    "name": student["name"],
                    "branch": student["branch"],
                    "cgpa": student["cgpa"],
                    "backlogs": student["backlogs"],
                    "batch": student["batch"]
                })

    os.makedirs("output", exist_ok=True)

    # Write job_summary.csv
    with open("output/job_summary.csv", "w", newline="", encoding="utf-8") as file:
        fields = [
            "job_id", "company", "branches", "min_cgpa",
            "max_backlogs", "batch", "summary", "status"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(job_summary_rows)

    # Write eligible.csv
    with open("output/eligible.csv", "w", newline="", encoding="utf-8") as file:
        fields = [
            "job_id", "company", "roll", "name", "branch",
            "cgpa", "backlogs", "batch"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(eligible_rows)

    # Write review.csv
    with open("output/review.csv", "w", newline="", encoding="utf-8") as file:
        fields = ["job_id", "company", "post", "reason"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)

    # Write summary.txt
    with open("output/summary.txt", "w", encoding="utf-8") as file:
        file.write("PLACEMENT ELIGIBILITY CHECKER - SUMMARY\n")
        file.write("=======================================\n")
        file.write(f"Total jobs: {len(job_results)}\n")
        file.write(f"Total students: {len(students)}\n")
        file.write(f"Jobs matched: {len(job_results) - len(review_rows)}\n")
        file.write(f"Jobs needing review: {len(review_rows)}\n")
        file.write(f"Eligible student-job matches: {len(eligible_rows)}\n")
        file.write("\nIMPORTANT: These are draft eligibility results.\n")
        file.write("A placement officer must check every draft list before sharing.\n")

    return len(eligible_rows), len(review_rows)


# Main program
def main():
    print("====================================")
    print("   PLACEMENT ELIGIBILITY CHECKER")
    print("====================================")

    try:
        jobs = load_jobs()
        students = load_students()
    except (OSError, KeyError, ValueError) as error:
        print("Error reading CSV files:", error)
        return

    print("Jobs loaded:", len(jobs))
    print("Students loaded:", len(students))

    job_results = []

    print("\nStarting AI criteria extraction...\n")

    for index, job in enumerate(jobs, start=1):
        print(f"Processing job {index}/{len(jobs)}: {job['company']}")

        criteria = read_criteria(job["post"])
        criteria = check_facts(job["post"], criteria)
        status = job_status(criteria)

        job_results.append({
            "job": job,
            "criteria": criteria,
            "status": status
        })

        print("Status:", status)
        print("Criteria:", criteria)
        print("------------------------------------")

    eligible_count, review_count = write_reports(job_results, students)

    print("\n====================================")
    print("          PROJECT COMPLETED")
    print("====================================")
    print("Eligible student-job matches:", eligible_count)
    print("Jobs needing manual review:", review_count)
    print("\nFour reports saved inside the output folder:")
    print("1. job_summary.csv")
    print("2. eligible.csv")
    print("3. review.csv")
    print("4. summary.txt")

    print("\nIMPORTANT:")
    print("These are draft results.")
    print("A placement officer must review the eligibility list")
    print("before it is shared with anyone.")


if __name__ == "__main__":
    main()



