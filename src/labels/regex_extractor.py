"""Regex/lexicon-based label extraction from multilingual reports."""

import re
from src.config import LABEL_COLUMNS
from src.labels.extractor import LabelDict


POSITIVE_PATTERNS: dict[str, list[str]] = {
    "ACL": [
        r"ACL\s+(?:tear|rupture|injury|disruption)",
        r"anterior\s+cruciate\s+ligament\s+(?:tear|rupture|injury|disruption)",
        r"rotura\s+(?:completa|parcial)\s+(?:del\s+)?ligamento\s+cruzado\s+anterior",
        r"LCA\s+(?:tear|rupture|Verletzung|ruptura)",
        r"rupture\s+(?:du\s+)?ligament\s+croisé\s+antérieur",
        r"vorderes\s+kreuzband\s+(?:ruptur|verletzung|riss)",
    ],
    "MCL": [
        r"MCL\s+(?:tear|rupture|injury|sprain)",
        r"medial\s+collateral\s+ligament\s+(?:tear|rupture|injury|sprain)",
        r"rotura\s+(?:del\s+)?ligamento\s+colateral\s+medial",
        r"rupture\s+(?:du\s+)?ligament\s+collatéral\s+médial",
        r"mediales\s+seitenband\s+(?:ruptur|verletzung|riss)",
    ],
    "Medial Meniscus": [
        r"medial\s+meniscus\s+(?:tear|rupture|injury|damage)",
        r"meniscus\s+medialis\s+(?:tear|rupture|injury)",
        r"rotura\s+(?:del\s+)?menisco\s+(?:interno|medial)",
        r"rotura\s+de\s+menisco\s+interno",
        r"ménisque\s+médial\s+(?:déchirure|lésion|rupture)",
        r"déchirure\s+.*ménisque\s+médial",
        r"meniskus\s+medialis\s+(?:riss|ruptur|verletzung)",
        r"riss\s+.*meniskus\s+medial",
        r"meniscal\s+tear.*medial",
        r"medial.*meniscal\s+tear",
        r"meniscus.*tear.*medial",
        r"medial.*meniscus.*tear",
    ],
    "Lateral Meniscus": [
        r"lateral\s+meniscus\s+(?:tear|rupture|injury|damage)",
        r"meniscus\s+lateralis\s+(?:tear|rupture|injury)",
        r"rotura\s+(?:del\s+)?menisco\s+externo",
        r"ménisque\s+latéral\s+(?:déchirure|lésion|rupture)",
        r"déchirure\s+.*ménisque\s+latéral",
        r"meniskus\s+lateralis\s+(?:riss|ruptur|verletzung)",
        r"riss\s+.*meniskus\s+lateral",
        r"meniscal\s+tear.*lateral",
        r"lateral.*meniscal\s+tear",
        r"meniscus.*tear.*lateral",
        r"lateral.*meniscus.*tear",
    ],
    "Medial OA": [
        r"medial\s+(?:compartment\s+)?(?:osteoarthritis|OA|arthrosis|cartilage\s+loss)",
        r"femorotibial\s+medial\s+(?:osteoarthritis|OA|arthrosis)",
        r"artrosis\s+(?:femorotibial\s+)?medial",
        r"arthrose\s+(?:fémoro-tibiale\s+)?médiale",
        r"mediale\s+arthrose",
        r"gonarthrose\s+médiale",
    ],
    "Lateral OA": [
        r"lateral\s+(?:compartment\s+)?(?:osteoarthritis|OA|arthrosis|cartilage\s+loss)",
        r"femorotibial\s+lateral\s+(?:osteoarthritis|OA|arthrosis)",
        r"artrosis\s+(?:femorotibial\s+)?lateral",
        r"arthrose\s+(?:fémoro-tibiale\s+)?latérale",
        r"laterale\s+arthrose",
    ],
    "PF OA": [
        r"(?:patellofemoral|patello-femoral|retropatellar)\s+(?:osteoarthritis|OA|arthrosis)",
        r"artrosis\s+(?:patelofemoral|retropatelar)",
        r"arthrose\s+(?:fémoro-patellaire|patello-fémorale)",
        r"patellofemorale\s+arthrose",
        r"chondropathie\s+retropatellaire",
        r"chondropatía\s+retropatelar",
    ],
    "Effusion": [
        r"(?:joint\s+)?effusion",
        r"(?:moderate|large|massive)\s+(?:joint\s+)?(?:fluid|effusion)",
        r"derrame\s+(?:articular|de\s+líquido)?",
        r"ergot\s+(?:articulaire|de\s+genou)?",
        r"épanchement\s+(?:articulaire)?",
        r"gelenkerguss",
        r"erguss\s+im\s+knie",
        r"liquid(?:o)?\s+(?:articular|en\s+la\s+articulación)",
    ],
    "Synovitis": [
        r"synovitis",
        r"synovite",
        r"sinovitis",
        r"synovialitis",
        r"inflammation\s+(?:of\s+the\s+)?synov",
        r"inflamación\s+sinovial",
    ],
    "Baker's": [
        r"baker[']?s?\s+cyst",
        r"kyste\s+(?:de\s+)?baker",
        r"quiste\s+(?:de\s+)?baker",
        r"baker[- ]?zyste",
        r"popliteal\s+cyst",
        r"quiste\s+poplíteo",
        r"zyste\s+poplitée",
    ],
    "Contusion": [
        r"(?:bone\s+)?contusion",
        r"(?:bone\s+)?bruise",
        r"(?:bone\s+)?marrow\s+edema",
        r"contusión\s+(?:ósea|de\s+médula)",
        r"œdème\s+(?:osseux|medullaire)",
        r"edema\s+(?:óseo|de\s+médula)",
        r"knochenmarködem",
        r"bone\s+marrow\s+oedema",
    ],
    "Fracture": [
        r"(?:acute\s+)?fracture",
        r"cortical\s+(?:break|fracture)",
        r"fractura\s+(?:aguda|cortical)",
        r"fracture\s+(?:aiguë|corticale)",
        r"fraktur",
        r"bruch",
        r"osteochondral\s+(?:fracture|fractura|fracture)",
    ],
}

NEGATIVE_PATTERNS: dict[str, list[str]] = {
    "ACL": [
        r"ACL\s+(?:is\s+)?(?:intact|normal|unremarkable|no\s+tear)",
        r"anterior\s+cruciate\s+ligament\s+(?:is\s+)?(?:intact|normal|unremarkable)",
        r"ligamento\s+cruzado\s+anterior\s+(?:intacto|normal|conservado)",
        r"ligament\s+croisé\s+antérieur\s+(?:intact|normal)",
        r"vorderes\s+kreuzband\s+(?:intakt|normal)",
    ],
    "MCL": [
        r"MCL\s+(?:intact|normal|unremarkable)",
        r"medial\s+collateral\s+ligament\s+(?:intact|normal|unremarkable)",
        r"ligamento\s+colateral\s+medial\s+(?:intacto|normal)",
    ],
    "Medial Meniscus": [
        r"medial\s+meniscus\s+(?:intact|normal|unremarkable|no\s+tear)",
        r"menisco\s+(?:interno|medial)\s+(?:intacto|normal|conservado)",
        r"ménisque\s+médial\s+(?:intact|normal)",
        r"meniscus\s+medialis\s+(?:intact|normal)",
    ],
    "Lateral Meniscus": [
        r"lateral\s+meniscus\s+(?:intact|normal|unremarkable|no\s+tear)",
        r"menisco\s+(?:externo|lateral)\s+(?:intacto|normal|conservado)",
        r"ménisque\s+latéral\s+(?:intact|normal)",
    ],
    "Effusion": [
        r"no\s+(?:significant\s+)?effusion",
        r"without\s+(?:significant\s+)?effusion",
        r"absence\s+(?:of\s+)?effusion",
        r"sin\s+derrame",
        r"sans\s+ergot",
        r"kein\s+erguss",
    ],
}


class RegexExtractor:
    """Extract labels using regex patterns on report text."""

    def __init__(self, case_insensitive: bool = True):
        flags = re.IGNORECASE if case_insensitive else 0
        self.positive_patterns = {
            label: [re.compile(p, flags) for p in patterns]
            for label, patterns in POSITIVE_PATTERNS.items()
        }
        self.negative_patterns = {
            label: [re.compile(p, flags) for p in patterns]
            for label, patterns in NEGATIVE_PATTERNS.items()
        }

    def extract(self, report: str) -> LabelDict:
        """Extract labels from a single report."""
        if not report or not isinstance(report, str):
            return {col: 0.5 for col in LABEL_COLUMNS}

        labels: LabelDict = {}
        for col in LABEL_COLUMNS:
            pos_patterns = self.positive_patterns.get(col, [])
            neg_patterns = self.negative_patterns.get(col, [])

            has_positive = any(p.search(report) for p in pos_patterns)
            has_negative = any(p.search(report) for p in neg_patterns)

            if has_negative:
                labels[col] = 0.0
            elif has_positive:
                labels[col] = 1.0
            else:
                labels[col] = 0.5

        return labels

    def extract_batch(self, reports: list[str]) -> list[LabelDict]:
        """Extract labels from multiple reports."""
        return [self.extract(report) for report in reports]
