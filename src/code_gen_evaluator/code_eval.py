"""
Code Generation Evaluation Pipeline
Evaluates generated code using:
1. Static Analysis (linting, validation)
2. LLM-as-a-Judge (using Ollama local model)
"""

import json
import re
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Ollama Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen3-coder:latest"


@dataclass
class EvalResult:
    """Evaluation result for a single code sample."""
    question: str
    generated_code: str
    ground_truth: str
    
    # Static Analysis Scores
    html_valid: bool = True
    css_valid: bool = True
    has_syntax_errors: bool = False
    linting_issues: List[str] = None
    security_issues: List[str] = None
    
    # LLM Judge Scores (0-10)
    correctness_score: float = 0.0
    quality_score: float = 0.0
    completeness_score: float = 0.0
    llm_feedback: str = ""
    
    # Overall
    overall_score: float = 0.0
    
    def __post_init__(self):
        if self.linting_issues is None:
            self.linting_issues = []
        if self.security_issues is None:
            self.security_issues = []


def generate_ollama(prompt: str, temperature: float = 0.2, top_p: float = 0.95) -> str:
    """Make HTTP request to Ollama API."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }
    
    response = requests.post(OLLAMA_URL, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()["response"]


class StaticAnalyzer:
    """Static code analysis for HTML/CSS/JS."""
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """Run all static analysis checks."""
        results = {
            "html_valid": self._check_html_validity(code),
            "css_valid": self._check_css_validity(code),
            "has_syntax_errors": False,
            "linting_issues": [],
            "security_issues": [],
        }
        
        # Check for common issues
        results["linting_issues"] = self._check_linting(code)
        results["security_issues"] = self._check_security(code)
        results["has_syntax_errors"] = len(results["linting_issues"]) > 0
        
        return results
    
    def _check_html_validity(self, code: str) -> bool:
        """Basic HTML validation."""
        issues = []
        
        # Check for unclosed tags
        open_tags = re.findall(r'<(\w+)[^>]*(?<!/)>', code)
        close_tags = re.findall(r'</(\w+)>', code)
        self_closing = ['img', 'br', 'hr', 'input', 'meta', 'link', 'area', 'base', 'col', 'embed', 'source', 'track', 'wbr']
        
        open_counts = {}
        for tag in open_tags:
            tag_lower = tag.lower()
            if tag_lower not in self_closing:
                open_counts[tag_lower] = open_counts.get(tag_lower, 0) + 1
        
        close_counts = {}
        for tag in close_tags:
            tag_lower = tag.lower()
            close_counts[tag_lower] = close_counts.get(tag_lower, 0) + 1
        
        # Check for mismatched tags
        for tag, count in open_counts.items():
            if close_counts.get(tag, 0) != count:
                return False
        
        return True
    
    def _check_css_validity(self, code: str) -> bool:
        """Basic CSS validation."""
        # Extract CSS from style tags or check raw CSS
        css_content = ""
        style_matches = re.findall(r'<style[^>]*>(.*?)</style>', code, re.DOTALL | re.IGNORECASE)
        if style_matches:
            css_content = "\n".join(style_matches)
        
        if not css_content:
            return True  # No CSS to validate
        
        # Check for balanced braces
        open_braces = css_content.count('{')
        close_braces = css_content.count('}')
        
        if open_braces != close_braces:
            return False
        
        return True
    
    def _check_linting(self, code: str) -> List[str]:
        """Check for common linting issues."""
        issues = []
        
        # Check for inline styles (prefer external/internal stylesheets)
        if re.search(r'style\s*=\s*["\'][^"\']+["\']', code):
            issues.append("Found inline styles - consider using CSS classes")
        
        # Check for missing alt attributes on images
        img_without_alt = re.findall(r'<img(?![^>]*alt\s*=)[^>]*>', code, re.IGNORECASE)
        if img_without_alt:
            issues.append(f"Found {len(img_without_alt)} image(s) without alt attribute")
        
        # Check for deprecated HTML tags
        deprecated_tags = ['center', 'font', 'strike', 'marquee', 'blink']
        for tag in deprecated_tags:
            if re.search(rf'<{tag}[^>]*>', code, re.IGNORECASE):
                issues.append(f"Found deprecated HTML tag: <{tag}>")
        
        # Check for missing viewport meta tag (for responsive design)
        if '<html' in code.lower() and 'viewport' not in code.lower():
            issues.append("Missing viewport meta tag for responsive design")
        
        # Check for !important in CSS (often a code smell)
        important_count = code.lower().count('!important')
        if important_count > 2:
            issues.append(f"Excessive use of !important ({important_count} occurrences)")
        
        return issues
    
    def _check_security(self, code: str) -> List[str]:
        """Check for potential security issues."""
        issues = []
        
        # Check for inline JavaScript event handlers
        js_handlers = ['onclick', 'onload', 'onerror', 'onmouseover', 'onfocus', 'onblur']
        for handler in js_handlers:
            if re.search(rf'{handler}\s*=', code, re.IGNORECASE):
                issues.append(f"Found inline JavaScript handler: {handler}")
        
        # Check for javascript: URLs
        if 'javascript:' in code.lower():
            issues.append("Found javascript: URL (potential XSS risk)")
        
        # Check for external scripts without integrity
        external_scripts = re.findall(r'<script[^>]*src\s*=\s*["\']https?://[^"\']+["\'][^>]*>', code, re.IGNORECASE)
        for script in external_scripts:
            if 'integrity=' not in script.lower():
                issues.append("External script without integrity attribute (SRI)")
        
        # Check for target="_blank" without rel="noopener"
        blank_links = re.findall(r'<a[^>]*target\s*=\s*["\']_blank["\'][^>]*>', code, re.IGNORECASE)
        for link in blank_links:
            if 'noopener' not in link.lower() and 'noreferrer' not in link.lower():
                issues.append("Found target='_blank' without rel='noopener' (security risk)")
        
        return issues


class LLMJudge:
    """LLM-as-a-Judge for code evaluation."""
    
    def __init__(self, model: str = MODEL_NAME):
        self.model = model
    
    def evaluate(self, question: str, generated_code: str, ground_truth: str) -> Dict[str, Any]:
        """Ask LLM to judge the generated code."""
        
        prompt = f"""You are a code review expert. Evaluate the following generated code against the requirements.

## Task/Question:
{question}

## Generated Code:
```
{generated_code}
```

## Expected Behavior (Ground Truth):
{ground_truth}

## Evaluation Criteria:
Rate each criterion from 0-10 and provide brief feedback.

1. **Correctness**: Does the code correctly implement what was asked?
2. **Quality**: Is the code well-structured, readable, and following best practices?
3. **Completeness**: Does the code fully address all requirements?

Respond in this exact JSON format only (no other text):
{{
    "correctness_score": <0-10>,
    "quality_score": <0-10>,
    "completeness_score": <0-10>,
    "feedback": "<brief overall feedback in 1-2 sentences>"
}}"""

        try:
            response = generate_ollama(prompt, temperature=0.1)
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    "correctness_score": float(result.get("correctness_score", 0)),
                    "quality_score": float(result.get("quality_score", 0)),
                    "completeness_score": float(result.get("completeness_score", 0)),
                    "llm_feedback": result.get("feedback", ""),
                }
        except Exception as e:
            print(f"LLM Judge error: {e}")
        
        return {
            "correctness_score": 0.0,
            "quality_score": 0.0,
            "completeness_score": 0.0,
            "llm_feedback": "Error during evaluation",
        }


class CodeGenEvaluator:
    """Main evaluator combining static analysis and LLM judge."""
    
    def __init__(self):
        self.static_analyzer = StaticAnalyzer()
        self.llm_judge = LLMJudge()
    
    def evaluate_sample(self, question: str, generated_code: str, ground_truth: str) -> EvalResult:
        """Evaluate a single code sample."""
        
        # Run static analysis
        static_results = self.static_analyzer.analyze(generated_code)
        
        # Run LLM judge
        llm_results = self.llm_judge.evaluate(question, generated_code, ground_truth)
        
        # Calculate overall score
        # Static analysis contributes 30%, LLM judge contributes 70%
        static_score = 10.0
        if not static_results["html_valid"]:
            static_score -= 3
        if not static_results["css_valid"]:
            static_score -= 3
        if static_results["linting_issues"]:
            static_score -= min(len(static_results["linting_issues"]) * 0.5, 2)
        if static_results["security_issues"]:
            static_score -= min(len(static_results["security_issues"]) * 1, 2)
        static_score = max(0, static_score)
        
        llm_avg = (
            llm_results["correctness_score"] + 
            llm_results["quality_score"] + 
            llm_results["completeness_score"]
        ) / 3
        
        overall = (static_score * 0.3) + (llm_avg * 0.7)
        
        return EvalResult(
            question=question,
            generated_code=generated_code[:500] + "..." if len(generated_code) > 500 else generated_code,
            ground_truth=ground_truth,
            html_valid=static_results["html_valid"],
            css_valid=static_results["css_valid"],
            has_syntax_errors=static_results["has_syntax_errors"],
            linting_issues=static_results["linting_issues"],
            security_issues=static_results["security_issues"],
            correctness_score=llm_results["correctness_score"],
            quality_score=llm_results["quality_score"],
            completeness_score=llm_results["completeness_score"],
            llm_feedback=llm_results["llm_feedback"],
            overall_score=round(overall, 2),
        )
    
    def evaluate_dataset(self, data_path: str) -> Dict[str, Any]:
        """Evaluate entire dataset from JSON file."""
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        print(f"\nEvaluating {len(data)} samples...\n")
        
        for i, sample in enumerate(data):
            print(f"[{i+1}/{len(data)}] Evaluating: {sample['question'][:50]}...")
            
            result = self.evaluate_sample(
                question=sample.get("question", ""),
                generated_code=sample.get("answer", ""),
                ground_truth=sample.get("ground_truth", ""),
            )
            results.append(asdict(result))
            
            print(f"  → Overall Score: {result.overall_score}/10")
            print(f"  → LLM Feedback: {result.llm_feedback[:100]}...")
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        return {
            "evaluation_date": datetime.now().isoformat(),
            "total_samples": len(results),
            "summary": summary,
            "detailed_results": results,
        }
    
    def _calculate_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not results:
            return {}
        
        scores = {
            "overall_score": [r["overall_score"] for r in results],
            "correctness_score": [r["correctness_score"] for r in results],
            "quality_score": [r["quality_score"] for r in results],
            "completeness_score": [r["completeness_score"] for r in results],
        }
        
        summary = {}
        for metric, values in scores.items():
            summary[metric] = {
                "mean": round(sum(values) / len(values), 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
            }
        
        # Count static analysis issues
        html_valid_count = sum(1 for r in results if r["html_valid"])
        css_valid_count = sum(1 for r in results if r["css_valid"])
        
        summary["static_analysis"] = {
            "html_valid_rate": f"{html_valid_count}/{len(results)}",
            "css_valid_rate": f"{css_valid_count}/{len(results)}",
            "total_linting_issues": sum(len(r["linting_issues"]) for r in results),
            "total_security_issues": sum(len(r["security_issues"]) for r in results),
        }
        
        return summary
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "evaluation_results"):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_path = output_path / f"code_eval_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {json_path}")
        
        # Save summary report
        report_path = output_path / f"code_eval_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("CODE GENERATION EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {results['evaluation_date']}\n")
            f.write(f"Total Samples: {results['total_samples']}\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("SUMMARY SCORES\n")
            f.write("-" * 40 + "\n")
            
            summary = results['summary']
            for metric in ['overall_score', 'correctness_score', 'quality_score', 'completeness_score']:
                if metric in summary:
                    f.write(f"\n{metric.replace('_', ' ').title()}:\n")
                    f.write(f"  Mean: {summary[metric]['mean']}/10\n")
                    f.write(f"  Min:  {summary[metric]['min']}/10\n")
                    f.write(f"  Max:  {summary[metric]['max']}/10\n")
            
            f.write("\n" + "-" * 40 + "\n")
            f.write("STATIC ANALYSIS\n")
            f.write("-" * 40 + "\n")
            sa = summary.get('static_analysis', {})
            f.write(f"HTML Valid: {sa.get('html_valid_rate', 'N/A')}\n")
            f.write(f"CSS Valid: {sa.get('css_valid_rate', 'N/A')}\n")
            f.write(f"Total Linting Issues: {sa.get('total_linting_issues', 0)}\n")
            f.write(f"Total Security Issues: {sa.get('total_security_issues', 0)}\n")
        
        print(f"Report saved to: {report_path}")
        
        return json_path, report_path


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate code generation results")
    parser.add_argument(
        "--input", "-i",
        default="src/llm-eval-ragas/code_gen_eval_data.json",
        help="Path to input JSON file with code samples"
    )
    parser.add_argument(
        "--output", "-o",
        default="evaluation_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CODE GENERATION EVALUATION")
    print("Using Static Analysis + LLM-as-a-Judge")
    print("=" * 60)
    
    evaluator = CodeGenEvaluator()
    
    # Run evaluation
    results = evaluator.evaluate_dataset(args.input)
    
    # Save results
    evaluator.save_results(results, args.output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    summary = results['summary']
    print(f"\nOverall Score: {summary['overall_score']['mean']}/10")
    print(f"Correctness:   {summary['correctness_score']['mean']}/10")
    print(f"Quality:       {summary['quality_score']['mean']}/10")
    print(f"Completeness:  {summary['completeness_score']['mean']}/10")


if __name__ == "__main__":
    main()
