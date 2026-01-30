import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DomainAdapterAgent:

    DOMAIN_KEYWORDS = {
        "fintech_payment": ["payment", "transaction", "fraud", "merchant", "amount", "card", "fintech", "pci"],
        "health": ["patient", "medical", "hospital", "icu", "diagnosis", "treatment", "health", "disease"],
        "environmental": ["climate", "carbon", "emission", "sustainability", "energy", "environment", "green"],
        "cybersecurity": ["security", "attack", "breach", "malware", "cyber", "threat"],
        "financial": ["credit", "loan", "bank", "financial", "money"],
        "web3": ["blockchain", "crypto", "nft", "defi", "token", "wallet", "smart contract"],
    }
    
    def __init__(self):
        self.detected_domain = None
        self.domain_context = {}
    
    def detect_domain(self, df: pd.DataFrame, dataset_name: str = "",
                     column_names: Optional[List[str]] = None) -> str:
        if column_names is None:
            column_names = list(df.columns)
        
        # Combine all text for keyword matching
        text_to_match = " ".join([
            dataset_name.lower(),
            " ".join([str(c).lower() for c in column_names]),
        ])
        
        scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_to_match)
            scores[domain] = score
        
        # Also check data patterns
        if len(df) > 0:
            for col in df.columns:
                col_lower = str(col).lower()
                for domain, keywords in self.DOMAIN_KEYWORDS.items():
                    if any(kw in col_lower for kw in keywords):
                        scores[domain] = scores.get(domain, 0) + 1
        
        if scores:
            detected = max(scores.items(), key=lambda x: x[1])[0]
            if scores[detected] > 0:
                self.detected_domain = detected
                logger.info(f"Detected domain: {detected} (score: {scores[detected]})")
                return detected
        
        self.detected_domain = "general"
        logger.info("No specific domain detected, using 'general'")
        return "general"
    
    def get_domain_context(self, domain: Optional[str] = None) -> str:
        """
        Get domain-specific context string for prompt generation.
        
        Args:
            domain: Domain name (if None, uses detected domain)
            
        Returns:
            Domain context description
        """
        if domain is None:
            domain = self.detected_domain or "general"
        
        context_map = {
            "fintech_payment": (
                "This is FinTech and payment/transaction data. Generate realistic payment "
                "transactions for fraud detection and payment solutions. Do not generate "
                "real card numbers, CVV, or PCI-DSS sensitive data. Preserve statistical "
                "patterns: amounts, time-of-day, merchant categories, and fraud/non-fraud balance."
            ),
            "health": (
                "This is healthcare/medical data. Generate realistic patient records, "
                "medical measurements, and clinical outcomes. Ensure medical plausibility: "
                "vital signs should be within physiological ranges, diagnoses should align "
                "with symptoms, and treatments should match conditions. Privacy and ethical "
                "considerations are critical."
            ),
            "environmental": (
                "This is environmental/sustainability data. Generate realistic climate metrics, "
                "energy consumption patterns, carbon emissions, and environmental indicators. "
                "Ensure physical plausibility: emissions should correlate with energy usage, "
                "temperature trends should be consistent, and sustainability metrics should "
                "reflect real-world patterns."
            ),
            "cybersecurity": (
                "This is cybersecurity/fraud detection data. Generate realistic security events, "
                "fraud patterns, attack signatures, and threat indicators. Ensure logical "
                "consistency: suspicious transactions should have anomalous features, attack "
                "patterns should be realistic, and security events should align with system logs."
            ),
            "financial": (
                "This is financial/credit data. Generate realistic financial transactions, "
                "credit scores, loan applications, and payment histories. Ensure economic "
                "plausibility: income should correlate with spending, credit scores should "
                "align with payment history, and transactions should follow realistic patterns."
            ),
            "web3": (
                "This is Web3/blockchain data. Generate realistic blockchain transactions, "
                "token transfers, smart contract interactions, and crypto wallet activities. "
                "Ensure blockchain consistency: addresses should be valid formats, transactions "
                "should follow blockchain rules, and token amounts should be non-negative."
            ),
            "general": (
                "This is general tabular data. Generate realistic, diverse samples that "
                "preserve statistical properties while introducing natural variation. "
                "Ensure feature consistency and label-feature alignment."
            ),
        }
        
        context = context_map.get(domain, context_map["general"])
        self.domain_context[domain] = context
        return context
    
    def adapt_constraints(self, df: pd.DataFrame, domain: Optional[str] = None) -> Dict[str, any]:
        """
        Adapt constraints based on domain knowledge.
        
        Args:
            df: Training dataframe
            domain: Domain name
            
        Returns:
            Domain-adapted constraint dictionary
        """
        if domain is None:
            domain = self.detected_domain or "general"
        
        constraints = {}
        
        # Domain-specific constraint adjustments
        if domain == "health":
            # Medical ranges for common vital signs
            health_ranges = {
                "age": (0, 120),
                "temperature": (95.0, 105.0),  # Fahrenheit
                "heart_rate": (30, 220),
                "blood_pressure_systolic": (70, 200),
                "blood_pressure_diastolic": (40, 120),
            }
            for col in df.columns:
                col_lower = str(col).lower()
                for key, (low, high) in health_ranges.items():
                    if key in col_lower:
                        constraints[col] = {"min": low, "max": high}
        
        elif domain == "environmental":
            # Environmental metric ranges
            env_ranges = {
                "temperature": (-50, 60),  # Celsius
                "humidity": (0, 100),  # Percentage
                "co2": (300, 1000),  # ppm
            }
            for col in df.columns:
                col_lower = str(col).lower()
                for key, (low, high) in env_ranges.items():
                    if key in col_lower:
                        constraints[col] = {"min": low, "max": high}
        
        elif domain == "financial":
            # Financial constraints
            for col in df.columns:
                col_lower = str(col).lower()
                if "amount" in col_lower or "balance" in col_lower or "income" in col_lower:
                    # Financial amounts should be non-negative (unless explicitly debt/expense)
                    if "debt" not in col_lower and "expense" not in col_lower:
                        constraints[col] = {"min": 0}
        
        return constraints
    
    def adapt_prompt_template(self, base_template: str, domain: Optional[str] = None) -> str:
        """
        Adapt the base prompt template with domain-specific instructions.
        
        Args:
            base_template: Base Pipeline V7 template
            domain: Domain name
            
        Returns:
            Domain-adapted template
        """
        if domain is None:
            domain = self.detected_domain or "general"
        
        domain_context = self.get_domain_context(domain)
        
        # Inject domain context into template
        adapted = base_template.replace(
            "Leverage your knowledge about {domain}",
            f"Leverage your knowledge about {domain_context}"
        )
        
        # Add domain-specific instructions
        domain_instructions = {
            "health": (
                "\n\nIMPORTANT DOMAIN-SPECIFIC RULES:\n"
                "- All medical values must be physiologically plausible\n"
                "- Diagnoses must align with symptoms and measurements\n"
                "- Patient privacy and ethical considerations apply\n"
            ),
            "environmental": (
                "\n\nIMPORTANT DOMAIN-SPECIFIC RULES:\n"
                "- Environmental metrics must follow physical laws\n"
                "- Correlations between emissions and energy usage should be realistic\n"
                "- Climate trends should be temporally consistent\n"
            ),
            "cybersecurity": (
                "\n\nIMPORTANT DOMAIN-SPECIFIC RULES:\n"
                "- Security events must follow realistic attack patterns\n"
                "- Fraud indicators should align with transaction anomalies\n"
                "- System logs should be temporally and logically consistent\n"
            ),
            "financial": (
                "\n\nIMPORTANT DOMAIN-SPECIFIC RULES:\n"
                "- Financial amounts must be economically plausible\n"
                "- Credit scores should correlate with payment history\n"
                "- Transactions should follow realistic spending patterns\n"
            ),
            "web3": (
                "\n\nIMPORTANT DOMAIN-SPECIFIC RULES:\n"
                "- Blockchain addresses must follow valid formats\n"
                "- Token amounts must be non-negative\n"
                "- Transactions must follow blockchain protocol rules\n"
            ),
        }
        
        if domain in domain_instructions:
            adapted += domain_instructions[domain]
        
        return adapted
