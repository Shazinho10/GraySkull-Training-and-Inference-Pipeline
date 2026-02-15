"""
Code Generation Evaluation Pipeline (Enhanced)
================================================
Evaluates fine-tuned code generation model outputs using:
  1. Static Analysis  (HTML validation, CSS validation, linting, security)
  2. LLM-as-a-Judge   (qwen3-coder:latest via Ollama)

Scoring:
  Overall = (Static Analysis Score × 0.30) + (LLM Judge Score × 0.70)

Outputs:
  - Per-sample detailed results
  - Aggregate statistics: mean, variance, std-dev, median, min/max
  - Score distribution histogram
"""

import json
import math
import re
import csv
import statistics
import requests
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import Counter

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DEFAULT_CONFIG_PATH = Path(__file__).parent / "code_eval_config.yaml"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-coder:latest"


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load YAML configuration."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f)
    # Sensible defaults if config missing
    return {
        "model": {"name": MODEL_NAME, "base_url": OLLAMA_URL, "temperature": 0.1, "top_p": 0.95, "timeout": 600},
        "scoring": {"static_analysis_weight": 0.30, "llm_judge_weight": 0.70},
        "output": {"output_dir": "evaluation_results"},
    }


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class StaticAnalysisResult:
    """Result of static analysis for a single sample."""
    html_valid: bool = True
    css_valid: bool = True
    linting_issues: List[str] = field(default_factory=list)
    security_issues: List[str] = field(default_factory=list)
    score: float = 10.0  # out of 10


@dataclass
class LLMJudgeResult:
    """Result from LLM-as-a-Judge for a single sample."""
    correctness_score: float = 0.0   # 0-10
    quality_score: float = 0.0       # 0-10
    completeness_score: float = 0.0  # 0-10
    feedback: str = ""
    score: float = 0.0               # weighted average 0-10


@dataclass
class SampleEvalResult:
    """Full evaluation result for one sample."""
    task_id: int = 0
    prompt: str = ""
    generated_code: str = ""
    ground_truth: str = ""

    # Component results
    static_analysis: StaticAnalysisResult = field(default_factory=StaticAnalysisResult)
    llm_judge: LLMJudgeResult = field(default_factory=LLMJudgeResult)

    # Final
    overall_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Truncate large code for readability
        if len(d["generated_code"]) > 600:
            d["generated_code"] = d["generated_code"][:600] + "…"
        return d


# ──────────────────────────────────────────────
# Ollama Client
# ──────────────────────────────────────────────

def call_ollama(
    prompt: str,
    model: str = MODEL_NAME,
    base_url: str = OLLAMA_URL,
    temperature: float = 0.1,
    top_p: float = 0.95,
    timeout: int = 600,
) -> str:
    """Make a synchronous HTTP request to Ollama generate endpoint."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "top_p": top_p},
    }
    resp = requests.post(base_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"]


# ──────────────────────────────────────────────
# Static Analyzer
# ──────────────────────────────────────────────

class StaticAnalyzer:
    """Static code analysis for generated HTML/CSS outputs."""

    SELF_CLOSING_TAGS = frozenset([
        "img", "br", "hr", "input", "meta", "link", "area",
        "base", "col", "embed", "source", "track", "wbr",
    ])
    DEPRECATED_TAGS = frozenset(["center", "font", "strike", "marquee", "blink", "big", "tt"])
    INLINE_HANDLERS = [
        "onclick", "ondblclick", "onmousedown", "onmouseup",
        "onmouseover", "onmouseout", "onmousemove",
        "onkeydown", "onkeyup", "onkeypress",
        "onload", "onerror", "onfocus", "onblur",
        "onchange", "onsubmit", "onreset", "onselect",
    ]

    def __init__(self, config: Dict[str, Any] = None):
        self.cfg = (config or {}).get("static_analysis", {})

    # ── HTML Validation ──────────────────────
    def check_html_validity(self, code: str) -> Tuple[bool, List[str]]:
        """Detect unclosed or mismatched HTML tags."""
        issues: List[str] = []
        open_tags = re.findall(r"<(\w+)(?:\s[^>]*)?(?<!/)>", code)
        close_tags = re.findall(r"</(\w+)\s*>", code)

        open_counts: Dict[str, int] = {}
        for tag in open_tags:
            t = tag.lower()
            if t not in self.SELF_CLOSING_TAGS:
                open_counts[t] = open_counts.get(t, 0) + 1

        close_counts: Dict[str, int] = {}
        for tag in close_tags:
            t = tag.lower()
            close_counts[t] = close_counts.get(t, 0) + 1

        for t, cnt in open_counts.items():
            c_cnt = close_counts.get(t, 0)
            if c_cnt < cnt:
                issues.append(f"Unclosed <{t}> tag (opened {cnt}, closed {c_cnt})")
            elif c_cnt > cnt:
                issues.append(f"Extra closing </{t}> tag (opened {cnt}, closed {c_cnt})")

        for t, cnt in close_counts.items():
            if t not in open_counts:
                issues.append(f"Closing </{t}> without matching open tag")

        valid = len(issues) == 0
        return valid, issues

    # ── CSS Validation ───────────────────────
    def check_css_validity(self, code: str) -> Tuple[bool, List[str]]:
        """Validate CSS syntax: balanced braces, valid property patterns."""
        issues: List[str] = []
        css_blocks = re.findall(r"<style[^>]*>(.*?)</style>", code, re.DOTALL | re.IGNORECASE)
        if not css_blocks:
            return True, []

        css_content = "\n".join(css_blocks)

        # Balanced braces
        opens = css_content.count("{")
        closes = css_content.count("}")
        if opens != closes:
            issues.append(f"Unbalanced CSS braces ({{ {opens} vs }} {closes})")

        # Unclosed strings
        for quote in ['"', "'"]:
            if css_content.count(quote) % 2 != 0:
                issues.append(f"Unclosed {quote} string in CSS")

        # Missing semicolons (heuristic: property lines not ending with ; or { or } )
        lines = css_content.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if (
                ":" in stripped
                and not stripped.endswith(";")
                and not stripped.endswith("{")
                and not stripped.endswith("}")
                and not stripped.endswith(",")
                and not stripped.startswith("/*")
                and not stripped.startswith("//")
                and not stripped.startswith("@")
                and "{" not in stripped
                and "}" not in stripped
            ):
                # Check it's actually a property line
                if re.match(r"^[\w\-]+\s*:", stripped):
                    issues.append(f"Possible missing semicolon in CSS (line ~{i})")

        valid = len(issues) == 0
        return valid, issues

    # ── Linting ──────────────────────────────
    def check_linting(self, code: str) -> List[str]:
        """Detect style/quality issues."""
        issues: List[str] = []

        # Inline styles
        inline_matches = re.findall(r'style\s*=\s*["\'][^"\']+["\']', code)
        if inline_matches:
            issues.append(f"Inline styles found ({len(inline_matches)} occurrence(s)) — prefer CSS classes")

        # Images without alt
        imgs_no_alt = re.findall(r"<img(?![^>]*\balt\s*=)[^>]*>", code, re.IGNORECASE)
        if imgs_no_alt:
            issues.append(f"{len(imgs_no_alt)} <img> tag(s) missing alt attribute")

        # Deprecated tags
        for tag in self.DEPRECATED_TAGS:
            if re.search(rf"<{tag}[\s>]", code, re.IGNORECASE):
                issues.append(f"Deprecated HTML tag <{tag}>")

        # Missing viewport meta (only when it looks like a full document)
        if re.search(r"<html", code, re.IGNORECASE) and "viewport" not in code.lower():
            issues.append("Missing viewport meta tag for responsive design")

        # Excessive !important
        important_count = len(re.findall(r"!important", code, re.IGNORECASE))
        threshold = self.cfg.get("linting", {}).get("important_threshold", 2)
        if important_count > threshold:
            issues.append(f"Excessive !important usage ({important_count} occurrences)")

        # Missing lang attribute on <html>
        html_tag = re.search(r"<html[^>]*>", code, re.IGNORECASE)
        if html_tag and "lang=" not in html_tag.group().lower():
            issues.append("Missing lang attribute on <html> tag")

        return issues

    # ── Security ─────────────────────────────
    def check_security(self, code: str) -> List[str]:
        """Detect potential security issues."""
        issues: List[str] = []

        # Inline JS event handlers
        for handler in self.INLINE_HANDLERS:
            if re.search(rf"\b{handler}\s*=", code, re.IGNORECASE):
                issues.append(f"Inline JS handler: {handler}")

        # javascript: URLs
        if re.search(r"javascript\s*:", code, re.IGNORECASE):
            issues.append("javascript: URL detected (XSS risk)")

        # External scripts without SRI
        ext_scripts = re.findall(
            r'<script[^>]*\bsrc\s*=\s*["\']https?://[^"\']+["\'][^>]*>',
            code, re.IGNORECASE,
        )
        for scr in ext_scripts:
            if "integrity=" not in scr.lower():
                issues.append("External <script> without integrity attribute (SRI)")

        # External stylesheets without SRI
        ext_links = re.findall(
            r'<link[^>]*\bhref\s*=\s*["\']https?://[^"\']+["\'][^>]*>',
            code, re.IGNORECASE,
        )
        for link in ext_links:
            if 'rel="stylesheet"' in link.lower() or "rel='stylesheet'" in link.lower():
                if "integrity=" not in link.lower():
                    issues.append("External stylesheet without integrity attribute (SRI)")

        # target="_blank" without rel="noopener"
        blank_links = re.findall(
            r'<a[^>]*target\s*=\s*["\']_blank["\'][^>]*>',
            code, re.IGNORECASE,
        )
        for link in blank_links:
            if "noopener" not in link.lower() and "noreferrer" not in link.lower():
                issues.append('target="_blank" without rel="noopener noreferrer"')

        return issues

    # ── Main Entry ───────────────────────────
    def analyze(self, code: str) -> StaticAnalysisResult:
        """Run full static analysis and compute a 0-10 score."""
        html_valid, html_issues = self.check_html_validity(code)
        css_valid, css_issues = self.check_css_validity(code)
        linting_issues = self.check_linting(code)
        security_issues = self.check_security(code)

        # Score starts at 10
        score = 10.0
        html_penalty = self.cfg.get("html_validation", {}).get("penalty", 3.0)
        css_penalty = self.cfg.get("css_validation", {}).get("penalty", 3.0)
        lint_per = self.cfg.get("linting", {}).get("per_issue_penalty", 0.5)
        lint_max = self.cfg.get("linting", {}).get("max_penalty", 2.0)
        sec_per = self.cfg.get("security", {}).get("per_issue_penalty", 1.0)
        sec_max = self.cfg.get("security", {}).get("max_penalty", 2.0)

        if not html_valid:
            score -= html_penalty
        if not css_valid:
            score -= css_penalty
        score -= min(len(linting_issues) * lint_per, lint_max)
        score -= min(len(security_issues) * sec_per, sec_max)
        score = max(0.0, round(score, 2))

        all_issues = linting_issues + security_issues
        if html_issues:
            all_issues = [f"[HTML] {i}" for i in html_issues] + all_issues
        if css_issues:
            all_issues = [f"[CSS] {i}" for i in css_issues] + all_issues

        return StaticAnalysisResult(
            html_valid=html_valid,
            css_valid=css_valid,
            linting_issues=linting_issues,
            security_issues=security_issues,
            score=score,
        )


# ──────────────────────────────────────────────
# LLM-as-a-Judge
# ──────────────────────────────────────────────

class LLMJudge:
    """Use an independent LLM (qwen3-coder:latest) as an automated judge."""

    JUDGE_PROMPT_TEMPLATE = """You are an expert code reviewer evaluating HTML/CSS code generated by an AI model.

## User Prompt (Task):
{prompt}

## Generated Code:
```html
{generated_code}
```

## Expected Behavior / Ground Truth:
{ground_truth}

## Evaluation Criteria
Rate EACH criterion from 0 to 10 (integers only). Be rigorous and differentiate quality.

1. **Correctness** (weight: HIGH)
   Does the code correctly and accurately implement the user's requirements?
   - 0-3: Major requirements missing or wrong
   - 4-6: Partially correct, some requirements met
   - 7-9: Mostly correct with minor issues
   - 10: Perfectly implements all requirements

2. **Quality** (weight: MEDIUM)
   Is the code well-structured, readable, and following best practices?
   - 0-3: Poor structure, no best practices
   - 4-6: Acceptable but could be improved
   - 7-9: Clean, well-organized code
   - 10: Exemplary code quality

3. **Completeness** (weight: HIGH)
   Does the code cover ALL requested features with no gaps?
   - 0-3: Significant features missing
   - 4-6: Core features present, extras missing
   - 7-9: Nearly complete
   - 10: All features fully implemented

Respond ONLY with valid JSON in this exact format (no markdown, no explanation):
{{"correctness_score": <0-10>, "quality_score": <0-10>, "completeness_score": <0-10>, "feedback": "<2-3 sentence assessment>"}}"""

    def __init__(self, config: Dict[str, Any] = None):
        model_cfg = (config or {}).get("model", {})
        self.model = model_cfg.get("name", MODEL_NAME)
        self.base_url = model_cfg.get("base_url", OLLAMA_URL)
        self.temperature = model_cfg.get("temperature", 0.1)
        self.top_p = model_cfg.get("top_p", 0.95)
        self.timeout = model_cfg.get("timeout", 600)

    def evaluate(self, prompt: str, generated_code: str, ground_truth: str) -> LLMJudgeResult:
        """Ask the LLM to judge one sample."""
        judge_prompt = self.JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt,
            generated_code=generated_code,
            ground_truth=ground_truth,
        )

        try:
            raw = call_ollama(
                judge_prompt,
                model=self.model,
                base_url=self.base_url,
                temperature=self.temperature,
                top_p=self.top_p,
                timeout=self.timeout,
            )

            # Strip potential <think>...</think> reasoning tags (qwen3-coder)
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

            # Extract JSON object
            json_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                correctness = float(min(max(data.get("correctness_score", 0), 0), 10))
                quality = float(min(max(data.get("quality_score", 0), 0), 10))
                completeness = float(min(max(data.get("completeness_score", 0), 0), 10))
                feedback = str(data.get("feedback", ""))

                # Weighted average: correctness=0.4, quality=0.2, completeness=0.4
                avg = correctness * 0.4 + quality * 0.2 + completeness * 0.4

                return LLMJudgeResult(
                    correctness_score=correctness,
                    quality_score=quality,
                    completeness_score=completeness,
                    feedback=feedback,
                    score=round(avg, 2),
                )
        except Exception as e:
            print(f"  ⚠ LLM Judge error: {e}")

        return LLMJudgeResult(feedback="Error during LLM evaluation")


# ──────────────────────────────────────────────
# Statistics Helpers
# ──────────────────────────────────────────────

def compute_statistics(values: List[float]) -> Dict[str, Any]:
    """Compute comprehensive descriptive statistics."""
    if not values:
        return {}
    n = len(values)
    mean = statistics.mean(values)
    variance = statistics.variance(values) if n > 1 else 0.0
    std_dev = statistics.stdev(values) if n > 1 else 0.0
    median = statistics.median(values)
    return {
        "n": n,
        "mean": round(mean, 4),
        "variance": round(variance, 4),
        "std_deviation": round(std_dev, 4),
        "median": round(median, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def compute_distribution(values: List[float], bins: List[float] = None) -> Dict[str, int]:
    """Bin values into ranges and return counts."""
    if bins is None:
        bins = [0, 2, 4, 6, 8, 10]
    dist: Dict[str, int] = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        label = f"{lo:.0f}-{hi:.0f}"
        dist[label] = sum(1 for v in values if lo <= v < hi)
    # Include exact max in last bin
    last_label = f"{bins[-2]:.0f}-{bins[-1]:.0f}"
    dist[last_label] += sum(1 for v in values if v == bins[-1])
    # Fix double-counting
    dist[last_label] -= sum(1 for v in values if bins[-2] <= v < bins[-1] and v == bins[-1])
    return dist


# ──────────────────────────────────────────────
# Main Evaluator
# ──────────────────────────────────────────────

class CodeGenEvaluator:
    """
    Evaluation pipeline combining:
      • Static Analysis  (30%)
      • LLM-as-a-Judge   (70%)
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.static_analyzer = StaticAnalyzer(self.config)
        self.llm_judge = LLMJudge(self.config)
        self.sa_weight = self.config.get("scoring", {}).get("static_analysis_weight", 0.30)
        self.llm_weight = self.config.get("scoring", {}).get("llm_judge_weight", 0.70)

    def evaluate_sample(
        self, task_id: int, prompt: str, generated_code: str, ground_truth: str
    ) -> SampleEvalResult:
        """Evaluate a single generated code sample."""
        sa = self.static_analyzer.analyze(generated_code)
        llm = self.llm_judge.evaluate(prompt, generated_code, ground_truth)
        overall = round(sa.score * self.sa_weight + llm.score * self.llm_weight, 2)

        return SampleEvalResult(
            task_id=task_id,
            prompt=prompt,
            generated_code=generated_code,
            ground_truth=ground_truth,
            static_analysis=sa,
            llm_judge=llm,
            overall_score=overall,
        )

    def evaluate_dataset(self, data_path: str) -> Dict[str, Any]:
        """Evaluate all samples from a JSON file."""
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total = len(data)
        print(f"\n{'='*65}")
        print(f"  CODE GENERATION EVALUATION — {total} samples")
        print(f"  Static Analysis (×{self.sa_weight}) + LLM Judge (×{self.llm_weight})")
        print(f"  Model: {self.config.get('model', {}).get('name', MODEL_NAME)}")
        print(f"{'='*65}\n")

        results: List[SampleEvalResult] = []
        for i, sample in enumerate(data):
            prompt = sample.get("question", sample.get("prompt", ""))
            code = sample.get("answer", sample.get("generated_code", ""))
            gt = sample.get("ground_truth", "")

            print(f"  [{i+1:>2}/{total}] {prompt[:60]}{'…' if len(prompt)>60 else ''}")

            result = self.evaluate_sample(i + 1, prompt, code, gt)
            results.append(result)

            sa = result.static_analysis
            llm = result.llm_judge
            print(f"        Static: {sa.score:>5.2f}/10  |  LLM: {llm.score:>5.2f}/10  |  Overall: {result.overall_score:>5.2f}/10")
            if sa.linting_issues:
                print(f"        Lint: {', '.join(sa.linting_issues[:2])}")
            if sa.security_issues:
                print(f"        Sec:  {', '.join(sa.security_issues[:2])}")
            if llm.feedback:
                print(f"        Judge: {llm.feedback[:120]}")
            print()

        # Aggregate
        report = self._build_report(results)
        return report

    def _build_report(self, results: List[SampleEvalResult]) -> Dict[str, Any]:
        """Build full evaluation report with statistics."""
        detailed = [r.to_dict() for r in results]

        # Extract score lists
        overall_scores = [r.overall_score for r in results]
        static_scores = [r.static_analysis.score for r in results]
        llm_scores = [r.llm_judge.score for r in results]
        correctness = [r.llm_judge.correctness_score for r in results]
        quality = [r.llm_judge.quality_score for r in results]
        completeness = [r.llm_judge.completeness_score for r in results]

        stats = {
            "overall_score": compute_statistics(overall_scores),
            "static_analysis_score": compute_statistics(static_scores),
            "llm_judge_score": compute_statistics(llm_scores),
            "correctness_score": compute_statistics(correctness),
            "quality_score": compute_statistics(quality),
            "completeness_score": compute_statistics(completeness),
        }

        distribution = {
            "overall_score": compute_distribution(overall_scores),
            "static_analysis_score": compute_distribution(static_scores),
            "llm_judge_score": compute_distribution(llm_scores),
        }

        # Static analysis summary
        html_valid_n = sum(1 for r in results if r.static_analysis.html_valid)
        css_valid_n = sum(1 for r in results if r.static_analysis.css_valid)
        total_lint = sum(len(r.static_analysis.linting_issues) for r in results)
        total_sec = sum(len(r.static_analysis.security_issues) for r in results)

        sa_summary = {
            "html_validity_rate": f"{html_valid_n}/{len(results)} ({html_valid_n/len(results)*100:.1f}%)",
            "css_validity_rate": f"{css_valid_n}/{len(results)} ({css_valid_n/len(results)*100:.1f}%)",
            "total_linting_issues": total_lint,
            "total_security_issues": total_sec,
        }

        return {
            "evaluation_meta": {
                "date": datetime.now().isoformat(),
                "model": self.config.get("model", {}).get("name", MODEL_NAME),
                "scoring_weights": {
                    "static_analysis": self.sa_weight,
                    "llm_judge": self.llm_weight,
                },
                "total_samples": len(results),
            },
            "statistics": stats,
            "score_distribution": distribution,
            "static_analysis_summary": sa_summary,
            "per_sample_results": detailed,
        }

    # ── Output Writers ───────────────────────
    def save_results(self, report: Dict[str, Any], output_dir: str = None):
        """Save evaluation report to JSON, CSV and TXT."""
        out = Path(output_dir or self.config.get("output", {}).get("output_dir", "evaluation_results"))
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON
        json_path = out / f"code_eval_results_{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  JSON  → {json_path}")

        # CSV
        csv_path = out / f"code_eval_results_{ts}.csv"
        self._write_csv(report, csv_path)
        print(f"  CSV   → {csv_path}")

        # TXT Report
        txt_path = out / f"code_eval_report_{ts}.txt"
        self._write_txt_report(report, txt_path)
        print(f"  TXT   → {txt_path}")

        return json_path, csv_path, txt_path

    def _write_csv(self, report: Dict[str, Any], path: Path):
        """Write per-sample results to CSV."""
        samples = report["per_sample_results"]
        if not samples:
            return
        fieldnames = [
            "task_id", "prompt", "overall_score",
            "static_score", "html_valid", "css_valid",
            "linting_issues_count", "security_issues_count",
            "llm_score", "correctness_score", "quality_score",
            "completeness_score", "llm_feedback",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in samples:
                writer.writerow({
                    "task_id": s["task_id"],
                    "prompt": s["prompt"][:100],
                    "overall_score": s["overall_score"],
                    "static_score": s["static_analysis"]["score"],
                    "html_valid": s["static_analysis"]["html_valid"],
                    "css_valid": s["static_analysis"]["css_valid"],
                    "linting_issues_count": len(s["static_analysis"]["linting_issues"]),
                    "security_issues_count": len(s["static_analysis"]["security_issues"]),
                    "llm_score": s["llm_judge"]["score"],
                    "correctness_score": s["llm_judge"]["correctness_score"],
                    "quality_score": s["llm_judge"]["quality_score"],
                    "completeness_score": s["llm_judge"]["completeness_score"],
                    "llm_feedback": s["llm_judge"]["feedback"][:200],
                })

    def _write_txt_report(self, report: Dict[str, Any], path: Path):
        """Write a human-readable evaluation report."""
        meta = report["evaluation_meta"]
        stats = report["statistics"]
        dist = report["score_distribution"]
        sa = report["static_analysis_summary"]
        samples = report["per_sample_results"]

        lines: List[str] = []
        w = 70
        lines.append("=" * w)
        lines.append("CODE GENERATION EVALUATION REPORT".center(w))
        lines.append("=" * w)
        lines.append(f"Date           : {meta['date']}")
        lines.append(f"Model          : {meta['model']}")
        lines.append(f"Total Samples  : {meta['total_samples']}")
        lines.append(f"Scoring Weights: Static {meta['scoring_weights']['static_analysis']:.0%}  "
                      f"| LLM Judge {meta['scoring_weights']['llm_judge']:.0%}")
        lines.append("")

        # ── Statistics ──
        lines.append("-" * w)
        lines.append("AGGREGATE STATISTICS")
        lines.append("-" * w)

        header = f"{'Metric':<25} {'Mean':>7} {'Var':>8} {'StdDev':>8} {'Median':>7} {'Min':>6} {'Max':>6}"
        lines.append(header)
        lines.append("-" * len(header))
        for metric_name, s in stats.items():
            label = metric_name.replace("_", " ").title()
            lines.append(
                f"{label:<25} {s['mean']:>7.2f} {s['variance']:>8.4f} "
                f"{s['std_deviation']:>8.4f} {s['median']:>7.2f} {s['min']:>6.2f} {s['max']:>6.2f}"
            )
        lines.append("")

        # ── Distribution ──
        lines.append("-" * w)
        lines.append("SCORE DISTRIBUTION")
        lines.append("-" * w)
        for metric_name, buckets in dist.items():
            label = metric_name.replace("_", " ").title()
            lines.append(f"\n  {label}:")
            for bucket, count in buckets.items():
                bar = "█" * count + "░" * (meta["total_samples"] - count)
                lines.append(f"    [{bucket:>5}] {bar} {count}")
        lines.append("")

        # ── Static Analysis ──
        lines.append("-" * w)
        lines.append("STATIC ANALYSIS SUMMARY")
        lines.append("-" * w)
        lines.append(f"  HTML Validity       : {sa['html_validity_rate']}")
        lines.append(f"  CSS Validity        : {sa['css_validity_rate']}")
        lines.append(f"  Total Lint Issues   : {sa['total_linting_issues']}")
        lines.append(f"  Total Security Issues: {sa['total_security_issues']}")
        lines.append("")

        # ── Per-sample ──
        lines.append("-" * w)
        lines.append("PER-SAMPLE RESULTS")
        lines.append("-" * w)
        for s in samples:
            lines.append(f"\n  Task #{s['task_id']:>2}: {s['prompt'][:60]}")
            lines.append(f"    Overall: {s['overall_score']:.2f}/10  "
                         f"| Static: {s['static_analysis']['score']:.2f}  "
                         f"| LLM: {s['llm_judge']['score']:.2f}")
            lines.append(f"    Correctness: {s['llm_judge']['correctness_score']:.0f}  "
                         f"| Quality: {s['llm_judge']['quality_score']:.0f}  "
                         f"| Completeness: {s['llm_judge']['completeness_score']:.0f}")
            if s["static_analysis"]["linting_issues"]:
                lines.append(f"    Lint: {'; '.join(s['static_analysis']['linting_issues'][:3])}")
            if s["static_analysis"]["security_issues"]:
                lines.append(f"    Sec:  {'; '.join(s['static_analysis']['security_issues'][:3])}")
            if s["llm_judge"]["feedback"]:
                lines.append(f"    Feedback: {s['llm_judge']['feedback'][:140]}")

        lines.append("\n" + "=" * w)
        lines.append("END OF REPORT".center(w))
        lines.append("=" * w)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ──────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Code Generation Evaluation — Static Analysis + LLM-as-a-Judge"
    )
    parser.add_argument(
        "-i", "--input",
        default=str(Path(__file__).parent / "code_gen_eval_tasks.json"),
        help="Path to JSON file with evaluation tasks",
    )
    parser.add_argument(
        "-o", "--output",
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "-c", "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    evaluator = CodeGenEvaluator(config)

    report = evaluator.evaluate_dataset(args.input)
    evaluator.save_results(report, args.output)

    # Print final summary
    stats = report["statistics"]
    meta = report["evaluation_meta"]
    print(f"\n{'='*65}")
    print("  EVALUATION SUMMARY")
    print(f"{'='*65}")
    print(f"  Samples        : {meta['total_samples']}")
    print(f"  Overall Score  : {stats['overall_score']['mean']:.2f} ± {stats['overall_score']['std_deviation']:.2f}")
    print(f"  Static Analysis: {stats['static_analysis_score']['mean']:.2f} ± {stats['static_analysis_score']['std_deviation']:.2f}")
    print(f"  LLM Judge      : {stats['llm_judge_score']['mean']:.2f} ± {stats['llm_judge_score']['std_deviation']:.2f}")
    print(f"  Correctness    : {stats['correctness_score']['mean']:.2f} ± {stats['correctness_score']['std_deviation']:.2f}")
    print(f"  Quality        : {stats['quality_score']['mean']:.2f} ± {stats['quality_score']['std_deviation']:.2f}")
    print(f"  Completeness   : {stats['completeness_score']['mean']:.2f} ± {stats['completeness_score']['std_deviation']:.2f}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
