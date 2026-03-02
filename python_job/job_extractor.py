"""
╔══════════════════════════════════════════════════════════════════╗
║          🔍 SMART JOB EXTRACTOR — Resume-Based Job Finder        ║
║  Platforms: LinkedIn · Indeed · Naukri · Google · Glassdoor      ║
║             Remotive · Arbeitnow · Jobicy · RemoteOK · Adzuna    ║
║  Cost: ₹0 (Completely Free) | Python 3.9+                        ║
╚══════════════════════════════════════════════════════════════════╝

SETUP (run once):
    pip install python-jobspy pdfplumber requests sentence-transformers scikit-learn pandas

USAGE:
    # Basic search (no resume):
    python job_extractor.py --query "python developer" --location "Bangalore"

    # With resume (AI match scoring):
    python job_extractor.py --resume my_resume.pdf --location "Hyderabad"

    # Full options:
    python job_extractor.py --resume cv.pdf --query "data scientist" --location "Pune" --top 20 --hours 72

    # Remote only (fastest):
    python job_extractor.py --resume cv.pdf --query "backend engineer" --no-jobspy

    # India only (no remote APIs):
    python job_extractor.py --resume cv.pdf --location "Delhi" --no-apis

ADZUNA (optional, best India data):
    Register free at https://developer.adzuna.com/signup
    Then set: ADZUNA_APP_ID and ADZUNA_APP_KEY below
"""

import argparse
import os
import re
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── OPTIONAL CONFIG ──────────────────────────────────────────────
ADZUNA_APP_ID  = ""   # Optional: get free key at developer.adzuna.com
ADZUNA_APP_KEY = ""   # Optional: paste your key here

# ─── JOB PLATFORMS FOR JOBSPY ─────────────────────────────────────
JOBSPY_SITES = ["indeed", "google", "naukri", "glassdoor"]
# Remove "linkedin" if you get blocked. Add "zip_recruiter" for US.

# ─── SKILLS DICTIONARY (for resume parsing) ───────────────────────
SKILLS_DB = [
    # Languages
    "python","java","javascript","typescript","c++","c#","golang","rust","kotlin","swift",
    "php","ruby","scala","r","matlab","bash","shell","perl","dart","lua","haskell",
    # Web
    "react","angular","vue","nextjs","nodejs","express","django","flask","fastapi","spring",
    "html","css","tailwind","bootstrap","graphql","rest","api","redux","webpack",
    # Data / ML / AI
    "machine learning","deep learning","nlp","computer vision","tensorflow","pytorch","keras",
    "scikit-learn","pandas","numpy","scipy","opencv","huggingface","transformers","llm",
    "openai","langchain","rag","generative ai","data science","data analysis","statistics",
    # Databases
    "sql","mysql","postgresql","mongodb","redis","elasticsearch","cassandra","dynamodb",
    "oracle","sqlite","neo4j","firebase","supabase","snowflake","bigquery",
    # Cloud / DevOps
    "aws","azure","gcp","docker","kubernetes","terraform","ansible","jenkins","gitlab ci",
    "github actions","ci/cd","linux","nginx","apache","microservices","serverless",
    # Tools
    "git","jira","confluence","figma","postman","swagger","sonarqube","splunk","grafana",
    # Methodologies
    "agile","scrum","kanban","tdd","bdd","oop","solid","design patterns",
    # Business
    "excel","tableau","power bi","sap","salesforce","erp","crm",
]


# ═══════════════════════════════════════════════════════════════════
# 1. RESUME PARSER
# ═══════════════════════════════════════════════════════════════════

def parse_resume(pdf_path: str) -> dict:
    """Extract text, skills, job title, experience from resume PDF."""
    try:
        import pdfplumber
    except ImportError:
        print("❌ Install pdfplumber: pip install pdfplumber")
        return {}

    print(f"📄 Parsing resume: {pdf_path}")
    text = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"❌ Resume parse error: {e}")
        return {}

    text_lower = text.lower()

    # Extract skills
    found_skills = sorted(set(s for s in SKILLS_DB if s in text_lower))

    # Guess job title from resume (first non-empty line or common patterns)
    title_guess = ""
    title_patterns = [
        r"(?:software|senior|junior|lead|principal|staff)\s+\w+\s+(?:engineer|developer|architect|analyst|scientist|manager)",
        r"(?:data|ml|ai|devops|cloud|full.?stack|front.?end|back.?end)\s+\w+",
        r"(?:product|project|program)\s+manager",
    ]
    for pattern in title_patterns:
        m = re.search(pattern, text_lower)
        if m:
            title_guess = m.group().strip()
            break

    # Years of experience
    exp_match = re.findall(r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)", text_lower)
    years_exp = max([int(x) for x in exp_match], default=0)

    resume = {
        "text": text,
        "skills": found_skills,
        "title_guess": title_guess,
        "years_exp": years_exp,
        "word_count": len(text.split()),
    }

    print(f"   ✅ Found {len(found_skills)} skills: {', '.join(found_skills[:8])}{'...' if len(found_skills)>8 else ''}")
    if title_guess:
        print(f"   🎯 Detected role: {title_guess.title()}")
    if years_exp:
        print(f"   📅 Experience: {years_exp}+ years")
    return resume


# ═══════════════════════════════════════════════════════════════════
# 2. JOB MATCHER (AI / Skill-Based)
# ═══════════════════════════════════════════════════════════════════

class JobMatcher:
    def __init__(self, resume: dict):
        self.resume = resume
        self.model = None
        self.resume_embedding = None
        self._load_model()

    def _load_model(self):
        """Load sentence transformer for semantic matching."""
        try:
            from sentence_transformers import SentenceTransformer
            print("🤖 Loading AI match model (first run downloads ~80MB)...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            resume_text = self.resume.get("text", "")[:2000]
            self.resume_embedding = self.model.encode(resume_text, convert_to_tensor=True)
            print("   ✅ AI model ready")
        except ImportError:
            print("⚠️  sentence-transformers not installed — using skill-only matching")
            print("    Install: pip install sentence-transformers")

    def score(self, job: dict) -> dict:
        """Score a job 0-100 against the resume."""
        jd = f"{job.get('title','')} {job.get('description','')} {job.get('company','')}".lower()
        resume_skills = self.resume.get("skills", [])

        # Skill overlap score (40%)
        jd_skills = [s for s in SKILLS_DB if s in jd]
        matched = [s for s in resume_skills if s in jd_skills]
        missing  = [s for s in jd_skills if s not in resume_skills]
        skill_score = (len(matched) / max(len(jd_skills), 1)) * 40 if jd_skills else 20

        # Semantic similarity score (50%)
        sem_score = 20  # default if no model
        if self.model and self.resume_embedding is not None:
            try:
                import torch
                from sentence_transformers import util
                jd_embedding = self.model.encode(jd[:500], convert_to_tensor=True)
                sim = util.pytorch_cos_sim(self.resume_embedding, jd_embedding).item()
                sem_score = max(0, min(sim, 1)) * 50
            except Exception:
                pass

        # Title match bonus (10%)
        title_bonus = 0
        title_guess = self.resume.get("title_guess", "").lower()
        job_title = job.get("title", "").lower()
        if title_guess and any(w in job_title for w in title_guess.split() if len(w) > 3):
            title_bonus = 10

        total = round(skill_score + sem_score + title_bonus)
        total = max(0, min(total, 100))

        if total >= 75:
            verdict = "🟢 Excellent"
        elif total >= 55:
            verdict = "🟡 Good"
        elif total >= 35:
            verdict = "🟠 Fair"
        else:
            verdict = "🔴 Poor"

        return {
            "score": total,
            "verdict": verdict,
            "matched_skills": ", ".join(matched[:8]),
            "missing_skills": ", ".join(missing[:8]),
        }


# ═══════════════════════════════════════════════════════════════════
# 3. JOBSPY SCRAPER (LinkedIn, Indeed, Naukri, Glassdoor, Google)
# ═══════════════════════════════════════════════════════════════════

def scrape_jobspy(query: str, location: str, hours: int = 72, results_per_site: int = 30) -> list:
    """Scrape jobs via python-jobspy (no API keys needed)."""
    try:
        from jobspy import scrape_jobs
    except ImportError:
        print("❌ Install jobspy: pip install python-jobspy")
        return []

    print(f"\n🕷️  JobSpy scraping: {', '.join(JOBSPY_SITES)}")
    jobs_out = []
    try:
        df = scrape_jobs(
            site_name=JOBSPY_SITES,
            search_term=query,
            location=location,
            results_wanted=results_per_site,
            hours_old=hours,
            country_indeed="india",
            linkedin_fetch_description=False,  # speeds up LinkedIn
            verbose=0,
        )
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                jobs_out.append({
                    "title": str(row.get("title", "")),
                    "company": str(row.get("company", "")),
                    "location": str(row.get("location", "")),
                    "is_remote": bool(row.get("is_remote", False)),
                    "job_type": str(row.get("job_type", "")),
                    "salary": str(row.get("min_amount", "Not disclosed")),
                    "description": str(row.get("description", ""))[:600],
                    "posted": str(row.get("date_posted", "")),
                    "url": str(row.get("job_url", "")),
                    "source": str(row.get("site", "JobSpy")),
                })
            print(f"   ✅ {len(jobs_out)} jobs scraped")
        else:
            print("   ⚠️  No results from JobSpy")
    except Exception as e:
        print(f"   ❌ JobSpy error: {e}")
    return jobs_out


# ═══════════════════════════════════════════════════════════════════
# 4. FREE API FETCHERS
# ═══════════════════════════════════════════════════════════════════

def _get(url: str, params: dict = None, timeout: int = 10) -> any:
    """Safe GET request."""
    try:
        import requests
        r = requests.get(url, params=params, timeout=timeout,
                         headers={"User-Agent": "JobExtractor/1.0"})
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"   ⚠️  API error {url[:50]}: {e}")
        return None


def fetch_remotive(query: str, limit: int = 50) -> list:
    """Remote tech jobs — no key needed."""
    print("   📡 Remotive...")
    data = _get("https://remotive.com/api/remote-jobs", {"search": query, "limit": limit})
    if not data:
        return []
    return [{
        "title": j.get("title", ""),
        "company": j.get("company_name", ""),
        "location": j.get("candidate_required_location", "Worldwide"),
        "is_remote": True,
        "job_type": j.get("job_type", ""),
        "salary": j.get("salary", "Not disclosed"),
        "description": re.sub(r"<[^>]+>", " ", j.get("description", ""))[:600],
        "posted": j.get("publication_date", "")[:10],
        "url": j.get("url", ""),
        "source": "Remotive",
    } for j in data.get("jobs", [])]


def fetch_arbeitnow(query: str, pages: int = 3) -> list:
    """Europe & remote tech roles — no key, no rate limit."""
    print("   📡 Arbeitnow...")
    results = []
    for page in range(1, pages + 1):
        data = _get("https://www.arbeitnow.com/api/job-board-api", {"q": query, "page": page})
        if not data:
            break
        for j in data.get("data", []):
            results.append({
                "title": j.get("title", ""),
                "company": j.get("company_name", ""),
                "location": j.get("location", "Remote"),
                "is_remote": j.get("remote", False),
                "job_type": "",
                "salary": "Not disclosed",
                "description": j.get("description", "")[:600],
                "posted": j.get("created_at", "")[:10],
                "url": j.get("url", ""),
                "source": "Arbeitnow",
            })
    return results


def fetch_jobicy(query: str, count: int = 50) -> list:
    """Remote jobs with salary data — no key needed."""
    print("   📡 Jobicy...")
    data = _get(f"https://jobicy.com/api/v2/remote-jobs", {"count": count, "search": query})
    if not data:
        return []
    return [{
        "title": j.get("jobTitle", ""),
        "company": j.get("companyName", ""),
        "location": j.get("jobGeo", "Remote"),
        "is_remote": True,
        "job_type": j.get("jobType", ""),
        "salary": j.get("annualSalaryMin", "Not disclosed"),
        "description": j.get("jobExcerpt", "")[:600],
        "posted": j.get("pubDate", "")[:10],
        "url": j.get("url", ""),
        "source": "Jobicy",
    } for j in data.get("jobs", [])]


def fetch_remoteok(query: str) -> list:
    """Startup & tech remote jobs — rate limit: 1/min."""
    print("   📡 RemoteOK (waiting 60s for rate limit)...")
    time.sleep(61)
    data = _get(f"https://remoteok.com/api?tag={query.replace(' ', '+')}")
    if not data:
        return []
    return [{
        "title": j.get("position", ""),
        "company": j.get("company", ""),
        "location": "Remote",
        "is_remote": True,
        "job_type": "",
        "salary": f"${j['salary_min']}-{j['salary_max']}" if j.get("salary_min") else "Not disclosed",
        "description": re.sub(r"<[^>]+>", " ", j.get("description", ""))[:600],
        "posted": j.get("date", "")[:10],
        "url": j.get("url", ""),
        "source": "RemoteOK",
    } for j in data if isinstance(j, dict) and j.get("position")]


def fetch_the_muse(query: str, pages: int = 2) -> list:
    """Corporate & startup roles — no key, 3600/hr."""
    print("   📡 The Muse...")
    results = []
    for page in range(1, pages + 1):
        data = _get("https://www.themuse.com/api/public/jobs",
                    {"query": query, "page": page, "level": "Mid Level,Senior Level,Entry Level", "fields": "jobs"})
        if not data:
            break
        for j in data.get("results", []):
            results.append({
                "title": j.get("name", ""),
                "company": j.get("company", {}).get("name", ""),
                "location": ", ".join(l.get("name", "") for l in j.get("locations", [])) or "Remote",
                "is_remote": any("remote" in l.get("name", "").lower() for l in j.get("locations", [])),
                "job_type": "",
                "salary": "Not disclosed",
                "description": j.get("contents", "")[:600],
                "posted": j.get("publication_date", "")[:10],
                "url": j.get("refs", {}).get("landing_page", ""),
                "source": "TheMuse",
            })
    return results


def fetch_adzuna(query: str, location: str, app_id: str, app_key: str, pages: int = 3) -> list:
    """India jobs with salary data — free key from developer.adzuna.com."""
    if not app_id or not app_key:
        return []
    print("   📡 Adzuna (India)...")
    results = []
    loc = location.split(",")[0].strip().lower() if location else "india"
    for page in range(1, pages + 1):
        data = _get(
            f"https://api.adzuna.com/v1/api/jobs/in/search/{page}",
            {"app_id": app_id, "app_key": app_key, "what": query, "where": loc,
             "results_per_page": 50, "content-type": "application/json"},
        )
        if not data:
            break
        for j in data.get("results", []):
            results.append({
                "title": j.get("title", ""),
                "company": j.get("company", {}).get("display_name", ""),
                "location": j.get("location", {}).get("display_name", "India"),
                "is_remote": False,
                "job_type": j.get("contract_type", ""),
                "salary": f"₹{j['salary_min']:.0f}-{j['salary_max']:.0f}" if j.get("salary_min") else "Not disclosed",
                "description": j.get("description", "")[:600],
                "posted": j.get("created", "")[:10],
                "url": j.get("redirect_url", ""),
                "source": "Adzuna",
            })
    return results


def fetch_all_api_jobs(query: str, location: str, skip_remoteok: bool = False) -> list:
    """Fetch from all free APIs."""
    print("\n🌐 Fetching from free APIs...")
    jobs = []
    jobs += fetch_remotive(query)
    time.sleep(1)
    jobs += fetch_arbeitnow(query)
    jobs += fetch_jobicy(query)
    jobs += fetch_the_muse(query)
    if not skip_remoteok:
        jobs += fetch_remoteok(query)
    else:
        print("   ⏭️  RemoteOK skipped (--skip-remoteok)")
    if ADZUNA_APP_ID and ADZUNA_APP_KEY:
        jobs += fetch_adzuna(query, location, ADZUNA_APP_ID, ADZUNA_APP_KEY)
    else:
        print("   ℹ️  Adzuna skipped (no key set — see top of script)")
    print(f"   ✅ {len(jobs)} jobs from free APIs")
    return jobs


# ═══════════════════════════════════════════════════════════════════
# 5. DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════

def deduplicate(jobs: list) -> list:
    """Remove duplicate jobs by title+company similarity."""
    seen = set()
    unique = []
    for job in jobs:
        key = f"{job.get('title','').lower().strip()[:40]}|{job.get('company','').lower().strip()[:30]}"
        if key not in seen and len(key) > 5:
            seen.add(key)
            unique.append(job)
    removed = len(jobs) - len(unique)
    if removed:
        print(f"   🧹 Removed {removed} duplicates → {len(unique)} unique jobs")
    return unique


# ═══════════════════════════════════════════════════════════════════
# 6. SAVE & DISPLAY
# ═══════════════════════════════════════════════════════════════════

def save_csv(jobs: list, filename: str):
    """Save jobs to CSV file."""
    try:
        import pandas as pd
        df = pd.DataFrame(jobs)
        df.to_csv(filename, index=False)
        print(f"\n💾 Saved: {filename} ({len(df)} jobs)")
    except ImportError:
        import csv
        with open(filename, "w", newline="", encoding="utf-8") as f:
            if jobs:
                writer = csv.DictWriter(f, fieldnames=jobs[0].keys())
                writer.writeheader()
                writer.writerows(jobs)
        print(f"\n💾 Saved: {filename} ({len(jobs)} jobs)")


def display_top(jobs: list, top: int = 15):
    """Pretty-print top jobs to console."""
    print(f"\n{'═'*72}")
    print(f"{'🏆 TOP JOB MATCHES':^72}")
    print(f"{'═'*72}")
    for i, job in enumerate(jobs[:top], 1):
        score_info = f"  {job.get('verdict','')} [{job.get('score','')}%]" if job.get("score") else ""
        print(f"\n#{i:2}  {job.get('title','N/A')[:55]}{score_info}")
        print(f"     🏢 {job.get('company','')[:40]}  📍 {job.get('location','')[:30]}")
        if job.get("salary") and job["salary"] != "Not disclosed":
            print(f"     💰 {job['salary']}")
        if job.get("matched_skills"):
            print(f"     ✅ Skills: {job['matched_skills'][:60]}")
        if job.get("missing_skills"):
            print(f"     📚 Learn: {job['missing_skills'][:60]}")
        print(f"     🌐 {job.get('source',''):<12}  📅 {job.get('posted','')[:10]}")
        if job.get("url"):
            print(f"     🔗 {job['url'][:70]}")
    print(f"\n{'═'*72}")


# ═══════════════════════════════════════════════════════════════════
# 7. MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🔍 Smart Job Extractor — Resume-Based Multi-Platform Job Finder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--resume", metavar="PATH", help="Path to resume PDF (enables AI scoring)")
    parser.add_argument("--query", metavar="TEXT", default="", help="Search keywords (auto-detected from resume if not given)")
    parser.add_argument("--location", metavar="CITY", default="India", help="Location filter (e.g., 'Bangalore', 'Hyderabad')")
    parser.add_argument("--top", type=int, default=15, help="Top N jobs to display (default: 15)")
    parser.add_argument("--hours", type=int, default=72, help="Jobs posted in last N hours (default: 72)")
    parser.add_argument("--output", metavar="NAME", default="jobs", help="Output CSV prefix (default: jobs)")
    parser.add_argument("--no-jobspy", action="store_true", help="Skip JobSpy (LinkedIn/Indeed/Naukri/Google)")
    parser.add_argument("--no-apis", action="store_true", help="Skip free REST APIs (Remotive/Jobicy/etc.)")
    parser.add_argument("--skip-remoteok", action="store_true", help="Skip RemoteOK (avoids 60s wait)")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  🔍 SMART JOB EXTRACTOR")
    print(f"  📍 Location : {args.location}")
    print(f"  ⏰ Posted   : Last {args.hours} hours")
    print("═"*60)

    # 1. Parse resume
    resume = {}
    matcher = None
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"❌ Resume not found: {args.resume}")
            sys.exit(1)
        resume = parse_resume(args.resume)
        if resume:
            matcher = JobMatcher(resume)

    # 2. Build query
    query = args.query
    if not query and resume.get("title_guess"):
        query = resume["title_guess"]
        print(f"🎯 Auto query from resume: '{query}'")
    if not query and resume.get("skills"):
        query = " ".join(resume["skills"][:4])
        print(f"🎯 Auto query from skills: '{query}'")
    if not query:
        query = "software engineer"
        print(f"⚠️  No query given — using default: '{query}'")

    # 3. Fetch jobs
    all_jobs = []

    if not args.no_jobspy:
        all_jobs += scrape_jobspy(query, args.location, hours=args.hours)

    if not args.no_apis:
        all_jobs += fetch_all_api_jobs(query, args.location, skip_remoteok=args.skip_remoteok)

    if not all_jobs:
        print("\n❌ No jobs found. Try:\n  --no-jobspy (use APIs only)\n  --no-apis (use JobSpy only)\n  Different --query or --location")
        sys.exit(0)

    # 4. Deduplicate
    all_jobs = deduplicate(all_jobs)
    print(f"\n📊 Total unique jobs fetched: {len(all_jobs)}")

    # 5. Score & rank (if resume provided)
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    if matcher:
        print("\n🤖 AI scoring jobs (may take ~30s)...")
        for i, job in enumerate(all_jobs):
            result = matcher.score(job)
            job.update(result)
            if (i + 1) % 50 == 0:
                print(f"   Scored {i+1}/{len(all_jobs)}...")

        all_jobs.sort(key=lambda j: j.get("score", 0), reverse=True)
        filename = f"{args.output}_ranked_{date_str}.csv"
    else:
        # Sort by posted date if no scoring
        all_jobs.sort(key=lambda j: j.get("posted", ""), reverse=True)
        filename = f"{args.output}_raw_{date_str}.csv"

    # 6. Display & save
    display_top(all_jobs, args.top)
    save_csv(all_jobs, filename)

    # Summary
    if matcher:
        excellent = sum(1 for j in all_jobs if j.get("score", 0) >= 75)
        good = sum(1 for j in all_jobs if 55 <= j.get("score", 0) < 75)
        print(f"\n📈 Match Summary:")
        print(f"   🟢 Excellent (75%+): {excellent} jobs")
        print(f"   🟡 Good (55-74%)   : {good} jobs")
        print(f"   📋 Total fetched   : {len(all_jobs)} jobs")
        print(f"\n💡 Tip: Check 'missing_skills' column to see what to learn next!")

    print(f"\n✅ Done! Open '{filename}' for full results.")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()

