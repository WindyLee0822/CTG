from pathlib import Path
import yaml

#AIzaSyDDtAyo35Gh1CR6Hc9EgevI-dhfR_T-Ljo
NEGATIVE_INF = -100000.0

PERSPECTIVE_API_KEY = 'AIzaSyDDtAyo35Gh1CR6Hc9EgevI-dhfR_T-Ljo'

PERSPECTIVE_API_ATTRIBUTES = {
    'TOXICITY'
}

PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)
