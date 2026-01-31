# import argparse
# import base64
# import json
# import os
# import time
# from pathlib import Path
# from typing import Dict, List, Optional, Set

# import requests
# from dotenv import load_dotenv

# # -------------------------------------------------------------------
# # SIMPLE LOGGER (replace with your logger if needed)
# # -------------------------------------------------------------------
# import logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logger = logging.getLogger(__name__)

# # -------------------------------------------------------------------
# # CONSTANTS
# # -------------------------------------------------------------------
# GITHUB_API_URL = "https://api.github.com"


# def rate_limit_sleep(reset_ts: Optional[int]) -> None:
#     if not reset_ts:
#         time.sleep(3)
#         return
#     now = int(time.time())
#     sleep_for = max(reset_ts - now + 2, 3)
#     logger.info(f"Rate limited. Sleeping {sleep_for}s.")
#     time.sleep(sleep_for)


# class GitHubScraper:
#     def __init__(
#         self,
#         token: Optional[str],
#         allow_licenses: Optional[Set[str]],
#         min_stars: int,
#         languages: Optional[List[str]],
#         topics: Optional[List[str]],
#         max_repos: int,
#         max_files_per_repo: int,
#         output_path: Path,
#     ):
#         self.token = token
#         self.allow_licenses = allow_licenses or {
#             "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "mpl-2.0"
#         }
#         self.min_stars = min_stars
#         self.languages = languages or ["JavaScript", "TypeScript", "Python"]
#         self.topics = topics or ["website", "template", "frontend"]
#         self.max_repos = max_repos
#         self.max_files_per_repo = max_files_per_repo
#         self.output_path = output_path

#         self.headers = {
#             "Accept": "application/vnd.github+json",
#             "User-Agent": "Website-LLM-Scraper/1.0",
#         }
#         if self.token:
#             self.headers["Authorization"] = f"Bearer {self.token}"

#         self.output_path.parent.mkdir(parents=True, exist_ok=True)

#     # ---------------------------------------------------------------
#     # REQUEST WRAPPER
#     # ---------------------------------------------------------------
#     def _request(self, method: str, url: str, params: Optional[Dict] = None) -> requests.Response:
#         for _ in range(3):
#             resp = requests.request(method, url, headers=self.headers, params=params)
#             if resp.status_code == 403 and "rate limit" in resp.text.lower():
#                 reset = resp.headers.get("X-RateLimit-Reset")
#                 rate_limit_sleep(int(reset) if reset else None)
#                 continue
#             if resp.status_code >= 500:
#                 time.sleep(2)
#                 continue
#             return resp
#         return resp

#     # ---------------------------------------------------------------
#     # SEARCH (BROAD, CORRECT)
#     # ---------------------------------------------------------------
#     def search_repositories(self, query: str) -> List[Dict]:
#         repos = []
#         page = 1

#         # BROAD QUERY — precision happens later
#         q = f'"{query}" in:readme in:description+stars:>={self.min_stars}'
#         logger.info(f"GitHub search query: {q}")

#         while len(repos) < self.max_repos and page <= 10:
#             resp = self._request(
#                 "GET",
#                 f"{GITHUB_API_URL}/search/repositories",
#                 {
#                     "q": q,
#                     "sort": "stars",
#                     "order": "desc",
#                     "per_page": 50,
#                     "page": page,
#                 },
#             )

#             if resp.status_code != 200:
#                 logger.warning(resp.text[:200])
#                 break

#             items = resp.json().get("items", [])
#             if not items:
#                 break

#             for repo in items:
#                 repos.append(repo)
#                 if len(repos) >= self.max_repos:
#                     break

#             logger.info(f"Collected {len(repos)} repos (page {page})")
#             page += 1

#         return repos

#     # ---------------------------------------------------------------
#     # FILTERS
#     # ---------------------------------------------------------------
#     def repo_is_allowed(self, repo: Dict) -> bool:
#         lic = repo.get("license") or {}
#         spdx = (lic.get("spdx_id") or "").lower()
#         return spdx in self.allow_licenses

#     def language_ok(self, repo: Dict) -> bool:
#         lang = repo.get("language")
#         return lang is None or lang in self.languages

#     def topics_ok(self, repo: Dict) -> bool:
#         return True


#     # ---------------------------------------------------------------
#     # TREE FETCH
#     # ---------------------------------------------------------------
#     def get_repo_tree(self, owner: str, repo: str, branch: str) -> List[Dict]:
#         branch_url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/branches/{branch}"
#         br = self._request("GET", branch_url)
#         if br.status_code != 200:
#             return []

#         tree_sha = br.json()["commit"]["commit"]["tree"]["sha"]
#         tree_url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/git/trees/{tree_sha}"
#         resp = self._request("GET", tree_url, {"recursive": 1})
#         if resp.status_code != 200:
#             return []

#         return resp.json().get("tree", [])

#     # ---------------------------------------------------------------
#     # FILE SELECTION
#     # ---------------------------------------------------------------
#     def interesting_path(self, path: str) -> bool:
#         return path.endswith((".html", ".css", ".js", ".tsx", ".jsx", ".md", ".json"))

#     def get_file_content(self, owner: str, repo: str, path: str, branch: str) -> Optional[str]:
#         url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}"
#         resp = self._request("GET", url, {"ref": branch})
#         if resp.status_code != 200:
#             return None

#         data = resp.json()
#         if isinstance(data, dict) and "content" in data:
#             try:
#                 return base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
#             except Exception:
#                 return None
#         return None

#     # ---------------------------------------------------------------
#     # MAIN SCRAPER
#     # ---------------------------------------------------------------
#     def scrape(self, query: str) -> None:
#         repos = self.search_repositories(query)
#         logger.info(f"Found {len(repos)} candidate repositories")

#         for repo in repos:
#             if not self.repo_is_allowed(repo):
#                 continue
#             if not self.language_ok(repo):
#                 continue
#             if not self.topics_ok(repo):
#                 continue

#             owner, name = repo["full_name"].split("/")
#             branch = repo.get("default_branch", "main")
#             tree = self.get_repo_tree(owner, name, branch)
#             if not tree:
#                 continue

#             files = [t["path"] for t in tree if t["type"] == "blob" and self.interesting_path(t["path"])]
#             files = files[: self.max_files_per_repo]

#             logger.info(f"{repo['full_name']} → {len(files)} files")

#             for path in files:
#                 content = self.get_file_content(owner, name, path, branch)
#                 if not content:
#                     continue

#                 record = {
#                     "source": "github",
#                     "repo": repo["full_name"],
#                     "stars": repo["stargazers_count"],
#                     "license": (repo.get("license") or {}).get("spdx_id"),
#                     "language": repo.get("language"),
#                     "path": path,
#                     "content": content,
#                 }

#                 with self.output_path.open("a", encoding="utf-8") as f:
#                     f.write(json.dumps(record, ensure_ascii=False) + "\n")


# # -------------------------------------------------------------------
# # CLI
# # -------------------------------------------------------------------
# def main():
#     load_dotenv()

#     parser = argparse.ArgumentParser("GitHub Website Dataset Scraper")
#     parser.add_argument("--query", default="website template")
#     parser.add_argument("--min-stars", type=int, default=0)
#     parser.add_argument("--max-repos", type=int, default=50)
#     parser.add_argument("--max-files-per-repo", type=int, default=200)
#     parser.add_argument("--output", default="scraping_output/github_websites.jsonl")
#     args = parser.parse_args()

#     scraper = GitHubScraper(
#         token=os.getenv("GITHUB_TOKEN"),
#         allow_licenses=None,
#         min_stars=args.min_stars,
#         languages=["JavaScript", "TypeScript", "Python"],
#         topics=["website", "template", "frontend"],
#         max_repos=args.max_repos,
#         max_files_per_repo=args.max_files_per_repo,
#         output_path=Path(args.output),
#     )

#     logger.info("Starting GitHub scraping...")
#     scraper.scrape(args.query)
#     logger.info("Scraping completed.")


# if __name__ == "__main__":
#     main()


#======================================
#=======================================

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests
from dotenv import load_dotenv
import logging

# -------------------------------------------------------------------
# LOGGER
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------
GITHUB_API_URL = "https://api.github.com"


# -------------------------------------------------------------------
# RATE LIMIT HANDLING
# -------------------------------------------------------------------
def handle_rate_limit(resp: requests.Response) -> None:
    remaining = resp.headers.get("X-RateLimit-Remaining")
    reset = resp.headers.get("X-RateLimit-Reset")

    if remaining is not None:
        remaining = int(remaining)

    if remaining is not None and remaining <= 5:
        reset_ts = int(reset) if reset else int(time.time()) + 60
        sleep_for = max(reset_ts - int(time.time()) + 5, 5)
        logger.warning(f"Rate limit low ({remaining}). Sleeping {sleep_for}s.")
        time.sleep(sleep_for)


# -------------------------------------------------------------------
# SCRAPER
# -------------------------------------------------------------------
class GitHubScraper:
    def __init__(
        self,
        token: Optional[str],
        allow_licenses: Optional[Set[str]],
        min_stars: int,
        languages: List[str],
        max_repos: int,
        max_files_per_repo: int,
        output_path: Path,
        state_path: Path,
    ):
        self.token = token
        self.allow_licenses = allow_licenses or {
            "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "mpl-2.0"
        }
        self.min_stars = min_stars
        self.languages = languages
        self.max_repos = max_repos
        self.max_files_per_repo = max_files_per_repo
        self.output_path = output_path
        self.state_path = state_path

        self.headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "GraySkull-Web-Scraper/1.0",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

        self.state = self._load_state()

    # ---------------------------------------------------------------
    # STATE MANAGEMENT (RESUME SAFE)
    # ---------------------------------------------------------------
    def _load_state(self) -> Dict:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as f:
                state = json.load(f)
            logger.info(f"Resuming from state: {len(state['processed_repos'])} repos done")
            return state
        return {
            "processed_repos": [],
            "processed_files": {},
        }

    def _save_state(self) -> None:
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    # ---------------------------------------------------------------
    # REQUEST WRAPPER
    # ---------------------------------------------------------------
    def _request(self, method: str, url: str, params: Optional[Dict] = None) -> requests.Response:
        for _ in range(3):
            resp = requests.request(method, url, headers=self.headers, params=params)
            handle_rate_limit(resp)

            if resp.status_code >= 500:
                time.sleep(2)
                continue
            return resp
        return resp

    # ---------------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------------
    def search_repositories(self, query: str) -> List[Dict]:
        repos = []
        page = 1
        q = f'"{query}" in:readme in:description stars:>={self.min_stars}'
        logger.info(f"GitHub search query: {q}")

        while len(repos) < self.max_repos and page <= 10:
            resp = self._request(
                "GET",
                f"{GITHUB_API_URL}/search/repositories",
                {
                    "q": q,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 50,
                    "page": page,
                },
            )

            if resp.status_code != 200:
                break

            items = resp.json().get("items", [])
            if not items:
                break

            for repo in items:
                if repo["full_name"] not in self.state["processed_repos"]:
                    repos.append(repo)
                if len(repos) >= self.max_repos:
                    break

            page += 1

        return repos

    # ---------------------------------------------------------------
    # FILTERS
    # ---------------------------------------------------------------
    def repo_is_allowed(self, repo: Dict) -> bool:
        lic = repo.get("license") or {}
        return (lic.get("spdx_id") or "").lower() in self.allow_licenses

    def language_ok(self, repo: Dict) -> bool:
        return repo.get("language") in self.languages or repo.get("language") is None

    # ---------------------------------------------------------------
    # TREE + FILES
    # ---------------------------------------------------------------
    def get_repo_tree(self, owner: str, repo: str, branch: str) -> List[Dict]:
        br = self._request("GET", f"{GITHUB_API_URL}/repos/{owner}/{repo}/branches/{branch}")
        if br.status_code != 200:
            return []

        sha = br.json()["commit"]["commit"]["tree"]["sha"]
        resp = self._request("GET", f"{GITHUB_API_URL}/repos/{owner}/{repo}/git/trees/{sha}", {"recursive": 1})
        return resp.json().get("tree", []) if resp.status_code == 200 else []

    def interesting_path(self, path: str) -> bool:
        return path.endswith((".html", ".css", ".js", ".jsx", ".tsx", ".md", ".json"))

    def get_file_content(self, owner: str, repo: str, path: str, branch: str) -> Optional[str]:
        resp = self._request(
            "GET",
            f"{GITHUB_API_URL}/repos/{owner}/{repo}/contents/{path}",
            {"ref": branch},
        )
        if resp.status_code != 200:
            return None
        try:
            return base64.b64decode(resp.json()["content"]).decode("utf-8", errors="ignore")
        except Exception:
            return None

    # ---------------------------------------------------------------
    # MAIN SCRAPE
    # ---------------------------------------------------------------
    def scrape(self, query: str) -> None:
        repos = self.search_repositories(query)

        for idx, repo in enumerate(repos, start=1):
            if len(self.state["processed_repos"]) >= self.max_repos:
                logger.info("Reached max repo limit.")
                break

            repo_name = repo["full_name"]
            logger.info(f"[{len(self.state['processed_repos'])+1}/{self.max_repos}] {repo_name}")

            if not self.repo_is_allowed(repo) or not self.language_ok(repo):
                self.state["processed_repos"].append(repo_name)
                self._save_state()
                continue

            owner, name = repo_name.split("/")
            branch = repo.get("default_branch", "main")

            tree = self.get_repo_tree(owner, name, branch)
            files = [
                t["path"] for t in tree
                if t["type"] == "blob" and self.interesting_path(t["path"])
            ][: self.max_files_per_repo]

            done_files = set(self.state["processed_files"].get(repo_name, []))

            for fidx, path in enumerate(files, start=1):
                if path in done_files:
                    continue

                logger.info(f"  → {path} ({fidx}/{len(files)})")
                content = self.get_file_content(owner, name, path, branch)
                if not content:
                    continue

                record = {
                    "source": "github",
                    "repo": repo_name,
                    "stars": repo["stargazers_count"],
                    "license": (repo.get("license") or {}).get("spdx_id"),
                    "language": repo.get("language"),
                    "path": path,
                    "content": content,
                }

                with self.output_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                done_files.add(path)
                self.state["processed_files"][repo_name] = list(done_files)
                self._save_state()

            self.state["processed_repos"].append(repo_name)
            self._save_state()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    load_dotenv()

    parser = argparse.ArgumentParser("GitHub Website Dataset Scraper (Resume Safe)")
    parser.add_argument("--query", default="website template")
    parser.add_argument("--max-repos", type=int, default=45)
    parser.add_argument("--max-files-per-repo", type=int, default=100)
    parser.add_argument("--output", default="scraping_output/github_websites.jsonl")
    parser.add_argument("--state", default="scraping_output/github_state.json")
    args = parser.parse_args()

    scraper = GitHubScraper(
        token=os.getenv("GITHUB_TOKEN"),
        allow_licenses=None,
        min_stars=0,
        languages=["JavaScript", "TypeScript", "Python"],
        max_repos=args.max_repos,
        max_files_per_repo=args.max_files_per_repo,
        output_path=Path(args.output),
        state_path=Path(args.state),
    )

    logger.info("Starting scraper...")
    scraper.scrape(args.query)
    logger.info("Scraping completed.")


if __name__ == "__main__":
    main()
