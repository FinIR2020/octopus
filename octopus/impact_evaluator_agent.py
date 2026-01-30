import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ImpactEvaluatorAgent:

    IMPACT_METRICS = {
        "health": {
            "patients_helped": "Potential patients helped",
            "research_acceleration": "Research acceleration factor",
            "privacy_preserved": "Privacy-preserved patients",
            "cost_savings": "Cost savings (USD)",
        },
        "environmental": {
            "carbon_reduction": "Carbon reduction (tons)",
            "energy_saved": "Energy saved (kWh)",
            "policy_impact": "Policy impact score",
        },
        "financial": {
            "fraud_prevented": "Fraud cases prevented",
            "cost_savings": "Cost savings (USD)",
            "access_improved": "Access improved",
        },
        "education": {
            "students_helped": "Students helped",
            "learning_improvement": "Learning improvement %",
            "accessibility": "Accessibility gain",
        },
    }

    def __init__(self, domain: str = "health"):
        self.domain = domain
        self.impact_metrics = {}
    
    def evaluate_research_impact(
        self,
        original_size: int,
        synthetic_size: int,
        use_case: str = "clinical_trial"
    ) -> Dict[str, Any]:
        impact = {
            "data_availability": {
                "original": original_size,
                "synthetic": synthetic_size,
                "increase_factor": synthetic_size / original_size if original_size > 0 else 0,
            },
            "research_capabilities": [],
            "estimated_impact": {}
        }
        if self.domain == "health":
            if use_case == "clinical_trial":
                impact["research_capabilities"].append("Enable rare disease research")
                impact["research_capabilities"].append("Accelerate drug discovery")
                impact["research_capabilities"].append("Improve treatment protocols")
                impact["estimated_impact"] = {
                    "patients_helped": synthetic_size * 0.1,
                    "research_acceleration": min(synthetic_size / original_size, 5.0),
                    "privacy_preserved": original_size,
                    "cost_savings": original_size * 1000,
                }
            
            elif use_case == "epidemiology":
                impact["research_capabilities"].append("Enable large-scale population studies")
                impact["research_capabilities"].append("Improve disease surveillance")
                impact["research_capabilities"].append("Support public health policy")
                
                impact["estimated_impact"] = {
                    "population_covered": synthetic_size,
                    "policy_impact": "high",
                    "public_health_benefit": "significant",
                }
        
        elif self.domain == "environmental":
            impact["research_capabilities"].append("Enable climate modeling")
            impact["research_capabilities"].append("Support sustainability research")
            impact["research_capabilities"].append("Inform policy decisions")
            
            impact["estimated_impact"] = {
                "carbon_reduction": synthetic_size * 0.5,
                "policy_impact": "medium",
            }
        
        elif self.domain == "education":
            impact["research_capabilities"].append("Enable personalized learning research")
            impact["research_capabilities"].append("Improve educational outcomes")
            impact["research_capabilities"].append("Support educational equity")
            
            impact["estimated_impact"] = {
                "students_helped": synthetic_size,
                "learning_improvement": "15-25%",
                "accessibility": "improved",
            }
        
        self.impact_metrics = impact
        return impact
    
    def calculate_social_value(
        self,
        impact_metrics: Dict[str, Any],
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        if domain is None:
            domain = self.domain
        
        social_value = {
            "overall_score": 0.0,
            "components": {},
            "recommendations": []
        }
        
        estimated = impact_metrics.get("estimated_impact", {})
        if domain == "health":
            patients_helped = estimated.get("patients_helped", 0)
            research_accel = estimated.get("research_acceleration", 1.0)
            privacy_preserved = estimated.get("privacy_preserved", 0)
            
            social_value["components"] = {
                "health_impact": patients_helped * 10,
                "research_impact": (research_accel - 1) * 50,
                "privacy_impact": privacy_preserved * 5,
            }
            
            social_value["overall_score"] = sum(social_value["components"].values())
            
            if social_value["overall_score"] > 1000:
                social_value["recommendations"].append("High impact - suitable for clinical deployment")
            elif social_value["overall_score"] > 500:
                social_value["recommendations"].append("Medium impact - suitable for research use")
            else:
                social_value["recommendations"].append("Low impact - needs more data or better quality")
        
        elif domain == "environmental":
            carbon_reduction = estimated.get("carbon_reduction", 0)
            social_value["components"] = {
                "environmental_impact": carbon_reduction * 20,
            }
            social_value["overall_score"] = sum(social_value["components"].values())
        
        elif domain == "education":
            students_helped = estimated.get("students_helped", 0)
            social_value["components"] = {
                "educational_impact": students_helped * 5,
            }
            social_value["overall_score"] = sum(social_value["components"].values())
        
        return social_value
    
    def generate_impact_report(
        self,
        impact_metrics: Dict[str, Any],
        social_value: Dict[str, Any]
    ) -> str:
        report = f"""
# Social Impact Report

## Domain: {self.domain.upper()}

## Data Impact
- Original Data Size: {impact_metrics.get('data_availability', {}).get('original', 0)}
- Synthetic Data Size: {impact_metrics.get('data_availability', {}).get('synthetic', 0)}
- Increase Factor: {impact_metrics.get('data_availability', {}).get('increase_factor', 0):.2f}x

## Research Capabilities Enabled
"""
        
        for capability in impact_metrics.get("research_capabilities", []):
            report += f"- {capability}\n"
        
        report += "\n## Estimated Impact\n"
        for key, value in impact_metrics.get("estimated_impact", {}).items():
            report += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        report += f"\n## Social Value Score: {social_value.get('overall_score', 0):.1f}\n"
        
        report += "\n## Recommendations\n"
        for rec in social_value.get("recommendations", []):
            report += f"- {rec}\n"
        
        return report
    
    def suggest_use_cases(self, domain: Optional[str] = None) -> List[Dict[str, str]]:
        if domain is None:
            domain = self.domain
        
        use_cases = {
            "health": [
                {
                    "name": "Clinical Trial Design",
                    "description": "Design and optimize clinical trials using synthetic patient data",
                    "impact": "Accelerate drug development, reduce costs",
                },
                {
                    "name": "Rare Disease Research",
                    "description": "Enable research on rare diseases with limited real data",
                    "impact": "Improve treatment for underserved populations",
                },
                {
                    "name": "Medical Training",
                    "description": "Train medical AI models without exposing real patient data",
                    "impact": "Improve healthcare AI while protecting privacy",
                },
            ],
            "environmental": [
                {
                    "name": "Climate Modeling",
                    "description": "Generate synthetic climate data for modeling",
                    "impact": "Improve climate predictions and policy",
                },
                {
                    "name": "Sustainability Research",
                    "description": "Enable research on sustainable practices",
                    "impact": "Support environmental protection",
                },
            ],
            "education": [
                {
                    "name": "Personalized Learning",
                    "description": "Generate synthetic student data for personalized learning research",
                    "impact": "Improve educational outcomes",
                },
                {
                    "name": "Educational Equity",
                    "description": "Enable research on educational disparities",
                    "impact": "Promote educational equity",
                },
            ],
        }
        
        return use_cases.get(domain, [])
