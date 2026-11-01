"""User-requested card exclusion. Independent of geometry decisions."""
import re

def excluded_card(*texts):
    text=' '.join(str(t) for t in texts)
    return bool(re.search(r'\bzekrom\b',text.replace('_',' '),re.I) and
                re.search(r'\bsm[\s_/-]*138\b',text.replace('_',' '),re.I))
