"""Multimodal Bayesian Fusion module for Dermatology Diagnosis.
Combines CNN image classifier output probabilities with demographic priors (Age, Gender, Location)
using Bayes' theorem based on HAM10000 dataset statistics.
"""

import math
from typing import Dict, Any, Optional

# HAM10000 demographic classes:
# AKIEC, BCC, BKL, DF, MEL, NV, VASC

# Gaussian prior distributions for Age: P(Age | Class) = N(mu, sigma^2)
AGE_PRIORS: Dict[str, Dict[str, float]] = {
    "AKIEC": {"mu": 68.8, "sigma": 11.2},
    "BCC":   {"mu": 66.7, "sigma": 12.1},
    "BKL":   {"mu": 64.2, "sigma": 13.9},
    "DF":    {"mu": 45.3, "sigma": 15.8},
    "MEL":   {"mu": 59.6, "sigma": 14.8},
    "NV":    {"mu": 38.2, "sigma": 17.4},
    "VASC":  {"mu": 50.1, "sigma": 18.2},
}

# Gender priors: P(Gender | Class)
GENDER_PRIORS: Dict[str, Dict[str, float]] = {
    # Keys: "male", "female"
    "AKIEC": {"male": 0.63, "female": 0.37},
    "BCC":   {"male": 0.62, "female": 0.38},
    "BKL":   {"male": 0.54, "female": 0.46},
    "DF":    {"male": 0.45, "female": 0.55},
    "MEL":   {"male": 0.60, "female": 0.40},
    "NV":    {"male": 0.50, "female": 0.50},
    "VASC":  {"male": 0.48, "female": 0.52},
}

# Anatomical location priors: P(Location | Class)
LOCATION_PRIORS: Dict[str, Dict[str, float]] = {
    "AKIEC": {"back": 0.05, "trunk": 0.05, "extremity": 0.15, "hand_foot": 0.03, "head_neck": 0.70, "other": 0.02},
    "BCC":   {"back": 0.15, "trunk": 0.15, "extremity": 0.08, "hand_foot": 0.01, "head_neck": 0.60, "other": 0.01},
    "BKL":   {"back": 0.25, "trunk": 0.25, "extremity": 0.15, "hand_foot": 0.03, "head_neck": 0.30, "other": 0.02},
    "DF":    {"back": 0.10, "trunk": 0.10, "extremity": 0.70, "hand_foot": 0.03, "head_neck": 0.05, "other": 0.02},
    "MEL":   {"back": 0.35, "trunk": 0.20, "extremity": 0.30, "hand_foot": 0.05, "head_neck": 0.08, "other": 0.02},
    "NV":    {"back": 0.25, "trunk": 0.25, "extremity": 0.30, "head_neck": 0.10, "hand_foot": 0.05, "other": 0.05},
    "VASC":  {"back": 0.15, "trunk": 0.40, "extremity": 0.30, "head_neck": 0.10, "hand_foot": 0.03, "other": 0.02},
}

class MultimodalBayesianFusion:
    """Combines image logits/probabilities with age, gender, and location priors."""

    @staticmethod
    def map_body_location(ui_location: str) -> str:
        """Maps Streamlit UI body location string to prior category keys."""
        loc_map = {
            "Lưng": "back",
            "Ngực / Bụng": "trunk",
            "Vai": "back",
            "Mông / Bẹn": "trunk",
            "Cánh tay": "extremity",
            "Đùi": "extremity",
            "Cẳng chân": "extremity",
            "Bàn tay": "hand_foot",
            "Bàn chân": "hand_foot",
            "Đầu / Mặt": "head_neck",
            "Cổ": "head_neck",
            "Vị trí khác": "other"
        }
        return loc_map.get(ui_location, "other")

    @staticmethod
    def get_age_likelihood(age: float, mu: float, sigma: float) -> float:
        """Calculates Gaussian likelihood P(Age | Class)."""
        if sigma <= 0:
            return 1.0
        exponent = -((age - mu) ** 2) / (2 * (sigma ** 2))
        return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(exponent)

    def fuse(
        self,
        image_probs: Dict[str, float],
        age: Optional[float] = None,
        gender: Optional[str] = None,
        body_location: Optional[str] = None,
        lambda_val: float = 0.85
    ) -> Dict[str, float]:
        """Performs Bayesian fusion of vision probabilities and demographic priors.

        P(C_k | Image, Age, Gender, Location) proportional to:
            P(C_k | Image) * P(Age | C_k) * P(Gender | C_k) * P(Location | C_k)
        
        We interpolate between fused probability and raw image probability using lambda_val:
            Final_P = lambda * Image_P + (1 - lambda) * Fused_P
        """
        # If no demographic info is provided, return image probabilities as-is
        has_age = age is not None and age >= 0
        has_gender = gender in ["male", "female", "Nam", "Nữ"]
        has_location = body_location is not None and body_location != "--- Chọn Vị trí tổn thương ---"

        if not (has_age or has_gender or has_location):
            return image_probs

        # Standardize gender
        gender_key = None
        if gender in ["male", "Nam"]:
            gender_key = "male"
        elif gender in ["female", "Nữ"]:
            gender_key = "female"

        # Standardize location
        location_key = None
        if has_location:
            location_key = self.map_body_location(body_location)

        CLASS_PRIORS: Dict[str, float] = {
            "NV": 0.6705,
            "MEL": 0.1111,
            "BKL": 0.1097,
            "BCC": 0.0513,
            "AKIEC": 0.0326,
            "VASC": 0.0142,
            "DF": 0.0115
        }

        demographic_probs = {}
        total_demo = 0.0

        for cls in image_probs:
            prior_c = CLASS_PRIORS.get(cls, 1.0 / len(image_probs))
            likelihood = 1.0

            # 1. Age likelihood
            if has_age and cls in AGE_PRIORS:
                prior = AGE_PRIORS[cls]
                likelihood *= self.get_age_likelihood(age, prior["mu"], prior["sigma"])

            # 2. Gender likelihood
            if gender_key and cls in GENDER_PRIORS:
                likelihood *= GENDER_PRIORS[cls].get(gender_key, 0.5)

            # 3. Location likelihood
            if location_key and cls in LOCATION_PRIORS:
                likelihood *= LOCATION_PRIORS[cls].get(location_key, 0.05)

            p_c_d = prior_c * likelihood
            demographic_probs[cls] = p_c_d
            total_demo += p_c_d

        # Normalize P(C_i | D)
        if total_demo > 0:
            for cls in demographic_probs:
                demographic_probs[cls] /= total_demo
        else:
            # Fallback to class priors
            demographic_probs = CLASS_PRIORS.copy()

        # Multiplicative Bayesian Fusion: P(C_i | V, D) proportional to [P(C_i|V)]^lambda * [P(C_i|D)]^(1-lambda)
        eps = 1e-9
        final_probs = {}
        total_final = 0.0

        for cls in image_probs:
            p_v = max(image_probs[cls], eps)
            p_d = max(demographic_probs.get(cls, eps), eps)

            fused_val = (p_v ** lambda_val) * (p_d ** (1.0 - lambda_val))
            final_probs[cls] = fused_val
            total_final += fused_val

        # Re-normalize to guarantee sum to 1.0
        if total_final > 0:
            for cls in final_probs:
                final_probs[cls] /= total_final
        else:
            final_probs = image_probs.copy()

        return final_probs
