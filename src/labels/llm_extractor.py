"""LLM-based label extraction from radiology reports."""

import json
import os
from typing import Optional

from src.config import LABEL_COLUMNS, NOT_ADDRESSED_VALUE
from src.labels.extractor import LabelDict


SYSTEM_PROMPT = """You are a musculoskeletal radiologist assistant. Your task is to extract findings from knee MRI radiology reports.

For each of the 12 findings below, determine whether the report:
- Explicitly states the finding is PRESENT → output 1.0
- Explicitly states the finding is ABSENT → output 0.0
- Does NOT address the finding (neither confirms nor denies) → output 0.5

UNIVERSAL RULES (apply to all 12 findings):
1. EXPLICIT NEGATION = 0.0: "no effusion", "sin rotura", "geen afwijkingen", "intact ACL" → 0.0
2. UNMENTIONED = 0.0: If a finding is never mentioned at all, output 0.0 (radiologists report positives and stay silent on negatives)
3. HEDGED MENTIONS: "possible small tear", "suspicious for" → use clinical judgment (1.0 if likely present, 0.5 if genuinely uncertain)
4. IMPRESSION/CONCLUSION WEIGHT: The Impression or Conclusion section carries the most diagnostic weight. If it contradicts the body, follow the Impression.
5. CURRENT FINDINGS ONLY: Only count current, acute findings. Do NOT count chronic/post-operative changes (e.g., "post-operative ACL reconstruction" is not an ACL tear)
6. TRANSLATE MENTALLY: The report may be in any language (English, Spanish, French, German, Dutch, Bulgarian, Turkish, etc.). Translate mentally and extract findings.

FINDING-SPECIFIC CLINICAL RULES:

1. ACL (anterior cruciate ligament):
   - COUNT: partial or complete tear, rupture, sprain, injury, disruption, "ACL deficient"
   - COUNT: ruptured post-surgical graft (this is an ACL finding)
   - DO NOT COUNT: intact graft, "post-operative ACL reconstruction" (unless graft is ruptured)
   - DO NOT COUNT: mucoid degeneration without tear

2. MCL (medial collateral ligament):
   - COUNT: partial or complete tear, rupture, sprain, injury, "MCL deficient"
   - DO NOT COUNT: "MCL thickened" without tear mention

3. Medial Meniscus:
   - COUNT: tear (any type: horizontal, vertical, radial, complex, bucket-handle), rupture, injury, "meniscus deficient"
   - DO NOT COUNT: "meniscus degenerated" without tear, post-operative meniscectomy (unless re-tear mentioned)

4. Lateral Meniscus:
   - COUNT: tear (any type), rupture, injury, "meniscus deficient"
   - DO NOT COUNT: "meniscus degenerated" without tear, post-operative meniscectomy (unless re-tear mentioned)

5. Medial OA (medial compartment osteoarthritis):
   - COUNT: cartilage loss, chondral defect, osteoarthritis, "bone-on-bone", joint space narrowing, "medial compartment arthrosis"
   - DO NOT COUNT: "mild chondrosis" without functional impact (use judgment)

6. Lateral OA (lateral compartment osteoarthritis):
   - COUNT: cartilage loss, chondral defect, osteoarthritis, "bone-on-bone", joint space narrowing
   - DO NOT COUNT: "mild chondrosis" without functional impact (use judgment)

7. PF OA (patellofemoral osteoarthritis):
   - COUNT: cartilage loss, chondral defect, osteoarthritis, "bone-on-bone", joint space narrowing in patellofemoral compartment
   - DO NOT COUNT: "mild chondrosis" without functional impact (use judgment)

8. Effusion (joint effusion):
   - COUNT: effusion, "fluid in joint", "joint distended", "excess fluid"
   - DO NOT COUNT: "trace effusion" or "minimal effusion" if radiologist clearly considers it insignificant (use judgment — some annotators count trace, some don't)
   - JUDGMENT CALL: "small effusion", "moderate effusion" → 1.0

9. Synovitis (synovial inflammation):
   - COUNT: synovitis, synovial thickening, synovial enhancement
   - COUNT (broad definition — radiologists describe synovitis indirectly):
     * Hoffa fat pad impingement / Hoffa's syndrome
     * Plica syndrome
     * Friction/impingement syndromes with inflammatory component
     * "Synovial proliferation"
   - DO NOT COUNT: "mild synovial thickening" without clinical significance (use judgment)
   - NOTE: Synovitis is the hardest finding to phrase around — radiologists rarely write "synovitis" directly

10. Baker's cyst (popliteal cyst):
    - COUNT: Baker's cyst, popliteal cyst, "cyst in popliteal fossa"
    - DO NOT COUNT: "popliteal fluid" without cyst mention

11. Contusion (bone contusion / bone marrow edema):
    - COUNT: bone contusion, bone bruise, marrow edema with TRAUMATIC pattern
    - DO NOT COUNT: degenerative marrow edema (this is OA, not contusion)
    - DO NOT COUNT: "marrow signal abnormality" without traumatic context
    - JUDGMENT: Must have traumatic pattern (e.g., "lateral femoral condyle contusion" from pivot-shift injury)

12. Fracture:
    - COUNT: acute fracture, insufficiency fracture, avulsion fracture, stress fracture
    - COUNT: "fracture line", "cortical disruption", " trabecular disruption"
    - DO NOT COUNT: chronic non-union (unless acute on chronic), "old fracture"
    - NOTE: Avulsion fractures and insufficiency fractures count (these are often missed)

OUTPUT FORMAT:
Output ONLY a JSON object with exactly these 12 keys and float values (0.0, 0.5, or 1.0):
{
  "ACL": <value>,
  "MCL": <value>,
  "Medial Meniscus": <value>,
  "Lateral Meniscus": <value>,
  "Medial OA": <value>,
  "Lateral OA": <value>,
  "PF OA": <value>,
  "Effusion": <value>,
  "Synovitis": <value>,
  "Baker's": <value>,
  "Contusion": <value>,
  "Fracture": <value>
}

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
