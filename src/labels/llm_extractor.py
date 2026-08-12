"""LLM-based label extraction from radiology reports."""

import json
import os
from typing import Optional

from src.config import LABEL_COLUMNS, NOT_ADDRESSED_VALUE
from src.labels.extractor import LabelDict


SYSTEM_PROMPT = """You are a musculoskeletal radiologist assistant. Your task is to extract findings from knee MRI radiology reports.

For each of the following 12 findings, determine whether the report:
- Explicitly states the finding is PRESENT → output 1.0
- Explicitly states the finding is ABSENT → output 0.0
- Does NOT address the finding (neither confirms nor denies) → output 0.5

The 12 findings are:
1. ACL - anterior cruciate ligament tear/injury
2. MCL - medial collateral ligament tear/injury
3. Medial Meniscus - medial meniscus tear
4. Lateral Meniscus - lateral meniscus tear
5. Medial OA - medial compartment osteoarthritis/cartilage loss
6. Lateral OA - lateral compartment osteoarthritis/cartilage loss
7. PF OA - patellofemoral osteoarthritis/cartilage loss
8. Effusion - joint effusion/excess fluid
9. Synovitis - synovial inflammation
10. Baker's - Baker's cyst / popliteal cyst
11. Contusion - bone contusion / bone marrow edema
12. Fracture - acute fracture

The report may be in any language (English, Spanish, French, German, Dutch, Bulgarian, Turkish, etc.).
Translate mentally and extract the findings.

Output ONLY a JSON object with exactly these 12 keys and float values (0.0, 0.5, or 1.0).
No other text, no markdown, no explanation."""


class LLMExtractor:
    """Extract labels using an LLM (OpenAI API or compatible)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            client_kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def extract(self, report: str) -> LabelDict:
        """Extract labels from a single report using the LLM."""
        if not report or not isinstance(report, str):
            return {col: NOT_ADDRESSED_VALUE for col in LABEL_COLUMNS}

        user_prompt = f"Extract findings from this knee MRI report:\n\n{report}"

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=300,
                )

                content = response.choices[0].message.content.strip()
                content = content.replace("```json", "").replace("```", "").strip()

                result = json.loads(content)
                labels = {}
                for col in LABEL_COLUMNS:
                    val = result.get(col, NOT_ADDRESSED_VALUE)
                    if val not in (0.0, 0.5, 1.0):
                        val = NOT_ADDRESSED_VALUE
                    labels[col] = float(val)
                return labels

            except (json.JSONDecodeError, KeyError, Exception) as e:
                if attempt == self.max_retries - 1:
                    print(f"LLM extraction failed after {self.max_retries} attempts: {e}")
                    return {col: NOT_ADDRESSED_VALUE for col in LABEL_COLUMNS}

        return {col: NOT_ADDRESSED_VALUE for col in LABEL_COLUMNS}

    def extract_batch(
        self, reports: list[str], show_progress: bool = True
    ) -> list[LabelDict]:
        """Extract labels from multiple reports."""
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(reports, desc="LLM extraction")
            except ImportError:
                iterator = reports
        else:
            iterator = reports

        return [self.extract(report) for report in iterator]
