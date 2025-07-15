import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

class ThreatIntelligenceEngine:
    """
    Advanced threat intelligence engine for zero-day attack analysis
    """
    
    def __init__(self):
        self.threat_levels = {
            'CRITICAL': {'threshold': 0.9, 'color': 'red'},
            'HIGH': {'threshold': 0.7, 'color': 'orange'},
            'MEDIUM': {'threshold': 0.5, 'color': 'yellow'},
            'LOW': {'threshold': 0.3, 'color': 'blue'},
            'INFO': {'threshold': 0.0, 'color': 'green'}
        }
        
        self.attack_signatures = {
            'advanced_persistent_threat': {
                'indicators': ['long_duration', 'multiple_services', 'file_modifications'],
                'description': 'Sophisticated, long-term attack maintaining persistent access',
                'severity': 'CRITICAL',
                'countermeasures': ['Network segmentation', 'Advanced monitoring', 'Incident response']
            },
            'polymorphic_malware': {
                'indicators': ['variable_patterns', 'evasion_techniques', 'signature_changes'],
                'description': 'Self-modifying malware designed to evade traditional detection',
                'severity': 'HIGH',
                'countermeasures': ['Behavioral analysis', 'Heuristic detection', 'Sandboxing']
            },
            'ai_powered_attack': {
                'indicators': ['adaptive_behavior', 'ml_patterns', 'intelligent_evasion'],
                'description': 'Machine learning-driven attack that adapts to defensive measures',
                'severity': 'CRITICAL',
                'countermeasures': ['AI-based defense', 'Adversarial training', 'Dynamic analysis']
            },
            'supply_chain_attack': {
                'indicators': ['trusted_sources', 'legitimate_certificates', 'insider_access'],
                'description': 'Attack through compromised trusted third-party components',
                'severity': 'HIGH',
                'countermeasures': ['Supply chain monitoring', 'Code signing verification', 'Zero trust']
            }
        }
    
    def analyze_attack_pattern(self, attack_data: pd.DataFrame, model_predictions: np.ndarray, 
                             anomaly_scores: np.ndarray) -> Dict:
        """Analyze attack patterns and generate threat intelligence"""
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(attack_data),
            'detected_attacks': np.sum(model_predictions == 1),
            'missed_attacks': np.sum(model_predictions == 0),
            'detection_rate': np.mean(model_predictions == 1),
            'average_anomaly_score': np.mean(anomaly_scores),
            'threat_level': self._calculate_threat_level(anomaly_scores, model_predictions),
            'attack_characteristics': self._analyze_attack_characteristics(attack_data),
            'evasion_analysis': self._analyze_evasion_techniques(attack_data, model_predictions, anomaly_scores),
            'recommendations': self._generate_recommendations(model_predictions, anomaly_scores)
        }
        
        return analysis
    
    def _calculate_threat_level(self, anomaly_scores: np.ndarray, predictions: np.ndarray) -> str:
        """Calculate overall threat level based on detection results"""
        
        missed_rate = np.mean(predictions == 0)
        avg_score = np.mean(anomaly_scores)
        
        if missed_rate > 0.3 or avg_score < 0.3:
            return 'CRITICAL'
        elif missed_rate > 0.2 or avg_score < 0.5:
            return 'HIGH'
        elif missed_rate > 0.1 or avg_score < 0.7:
            return 'MEDIUM'
        elif missed_rate > 0.05:
            return 'LOW'
        else:
            return 'INFO'
    
    def _analyze_attack_characteristics(self, attack_data: pd.DataFrame) -> Dict:
        """Analyze key characteristics of attack patterns"""
        
        characteristics = {}
        
        # Temporal analysis
        if 'duration' in attack_data.columns:
            characteristics['duration_stats'] = {
                'mean': float(attack_data['duration'].mean()),
                'std': float(attack_data['duration'].std()),
                'min': float(attack_data['duration'].min()),
                'max': float(attack_data['duration'].max())
            }
        
        # Network traffic analysis
        if 'src_bytes' in attack_data.columns and 'dst_bytes' in attack_data.columns:
            characteristics['traffic_volume'] = {
                'avg_src_bytes': float(attack_data['src_bytes'].mean()),
                'avg_dst_bytes': float(attack_data['dst_bytes'].mean()),
                'total_bytes': float(attack_data['src_bytes'].sum() + attack_data['dst_bytes'].sum())
            }
        
        # Protocol distribution
        if 'protocol_type' in attack_data.columns:
            protocol_counts = attack_data['protocol_type'].value_counts()
            characteristics['protocol_distribution'] = protocol_counts.to_dict()
        
        # Service targeting
        if 'service' in attack_data.columns:
            service_counts = attack_data['service'].value_counts()
            characteristics['targeted_services'] = service_counts.head(5).to_dict()
        
        # Connection patterns
        if 'count' in attack_data.columns:
            characteristics['connection_patterns'] = {
                'avg_connections': float(attack_data['count'].mean()),
                'max_connections': float(attack_data['count'].max()),
                'connection_variance': float(attack_data['count'].var())
            }
        
        return characteristics
    
    def _analyze_evasion_techniques(self, attack_data: pd.DataFrame, predictions: np.ndarray, 
                                  anomaly_scores: np.ndarray) -> Dict:
        """Analyze evasion techniques used in attacks"""
        
        evasion_analysis = {}
        
        # Identify low-score attacks (potential evasion)
        low_score_threshold = 0.3
        potential_evasion = anomaly_scores < low_score_threshold
        
        if np.any(potential_evasion):
            evasion_samples = attack_data[potential_evasion]
            evasion_analysis['evasion_count'] = int(np.sum(potential_evasion))
            evasion_analysis['evasion_rate'] = float(np.mean(potential_evasion))
            
            # Analyze characteristics of evasion attempts
            if len(evasion_samples) > 0:
                evasion_analysis['evasion_characteristics'] = {}
                
                # Duration-based evasion
                if 'duration' in evasion_samples.columns:
                    evasion_analysis['evasion_characteristics']['duration_pattern'] = {
                        'avg_duration': float(evasion_samples['duration'].mean()),
                        'duration_variance': float(evasion_samples['duration'].var())
                    }
                
                # Size-based evasion
                if 'src_bytes' in evasion_samples.columns:
                    evasion_analysis['evasion_characteristics']['size_pattern'] = {
                        'avg_src_bytes': float(evasion_samples['src_bytes'].mean()),
                        'size_variance': float(evasion_samples['src_bytes'].var())
                    }
        
        # Missed attacks analysis
        missed_attacks = predictions == 0
        if np.any(missed_attacks):
            evasion_analysis['successful_evasion'] = {
                'count': int(np.sum(missed_attacks)),
                'rate': float(np.mean(missed_attacks)),
                'avg_anomaly_score': float(np.mean(anomaly_scores[missed_attacks]))
            }
        
        return evasion_analysis
    
    def _generate_recommendations(self, predictions: np.ndarray, anomaly_scores: np.ndarray) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        detection_rate = np.mean(predictions == 1)
        avg_score = np.mean(anomaly_scores)
        score_variance = np.var(anomaly_scores)
        
        # Detection rate recommendations
        if detection_rate < 0.8:
            recommendations.append("URGENT: Improve model training with more diverse attack patterns")
            recommendations.append("Consider implementing ensemble methods for better coverage")
        elif detection_rate < 0.9:
            recommendations.append("Fine-tune detection thresholds for better performance")
            recommendations.append("Add more training data from recent attack patterns")
        
        # Score analysis recommendations
        if avg_score < 0.5:
            recommendations.append("Review and adjust anomaly scoring mechanism")
            recommendations.append("Implement additional feature engineering")
        
        # Variance analysis recommendations
        if score_variance > 0.1:
            recommendations.append("Model shows inconsistent scoring - consider regularization")
            recommendations.append("Implement confidence intervals for anomaly scores")
        
        # Evasion mitigation recommendations
        missed_rate = np.mean(predictions == 0)
        if missed_rate > 0.1:
            recommendations.append("Implement anti-evasion techniques in model training")
            recommendations.append("Consider adversarial training approaches")
        
        # General security recommendations
        recommendations.extend([
            "Implement continuous monitoring for new attack patterns",
            "Regular model retraining with latest threat intelligence",
            "Deploy multiple detection models for comprehensive coverage"
        ])
        
        return recommendations
    
    def generate_threat_report(self, analysis: Dict) -> str:
        """Generate comprehensive threat intelligence report"""
        
        report = f"""
# ZERO-DAY THREAT INTELLIGENCE REPORT
Generated: {analysis['timestamp']}

## EXECUTIVE SUMMARY
Threat Level: {analysis['threat_level']}
Detection Rate: {analysis['detection_rate']:.1%}
Total Attacks Analyzed: {analysis['total_samples']}
Successful Detections: {analysis['detected_attacks']}
Missed Attacks: {analysis['missed_attacks']}

## ATTACK PATTERN ANALYSIS
Average Anomaly Score: {analysis['average_anomaly_score']:.3f}

### Key Characteristics:
"""
        
        if 'attack_characteristics' in analysis:
            chars = analysis['attack_characteristics']
            
            if 'duration_stats' in chars:
                report += f"- Attack Duration: {chars['duration_stats']['mean']:.1f}s (avg), {chars['duration_stats']['max']:.1f}s (max)\n"
            
            if 'traffic_volume' in chars:
                report += f"- Traffic Volume: {chars['traffic_volume']['total_bytes']/1024:.1f} KB total\n"
            
            if 'protocol_distribution' in chars:
                report += f"- Primary Protocols: {list(chars['protocol_distribution'].keys())[:3]}\n"
        
        report += "\n## EVASION ANALYSIS\n"
        if 'evasion_analysis' in analysis:
            evasion = analysis['evasion_analysis']
            
            if 'evasion_rate' in evasion:
                report += f"- Evasion Attempts: {evasion['evasion_rate']:.1%} of samples\n"
            
            if 'successful_evasion' in evasion:
                report += f"- Successful Evasion: {evasion['successful_evasion']['rate']:.1%}\n"
                report += f"- Evasion Score: {evasion['successful_evasion']['avg_anomaly_score']:.3f}\n"
        
        report += "\n## RECOMMENDATIONS\n"
        for i, rec in enumerate(analysis['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        return report
    
    def create_alert(self, threat_level: str, attack_type: str, details: Dict) -> Dict:
        """Create standardized threat alert"""
        
        alert = {
            'id': f"ZD-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{np.random.randint(1000, 9999)}",
            'timestamp': datetime.now().isoformat(),
            'threat_level': threat_level,
            'attack_type': attack_type,
            'severity': self.attack_signatures.get(attack_type, {}).get('severity', 'MEDIUM'),
            'description': self.attack_signatures.get(attack_type, {}).get('description', 'Unknown attack pattern'),
            'details': details,
            'countermeasures': self.attack_signatures.get(attack_type, {}).get('countermeasures', []),
            'status': 'ACTIVE'
        }
        
        return alert
    
    def calculate_risk_score(self, attack_characteristics: Dict, detection_performance: Dict) -> float:
        """Calculate overall risk score based on attack and detection characteristics"""
        
        base_risk = 0.5
        
        # Adjust based on detection performance
        detection_rate = detection_performance.get('detection_rate', 0.5)
        if detection_rate < 0.7:
            base_risk += 0.3
        elif detection_rate < 0.9:
            base_risk += 0.1
        
        # Adjust based on attack sophistication
        if 'evasion_analysis' in attack_characteristics:
            evasion_rate = attack_characteristics['evasion_analysis'].get('evasion_rate', 0)
            base_risk += evasion_rate * 0.2
        
        # Adjust based on attack volume
        if 'traffic_volume' in attack_characteristics:
            total_bytes = attack_characteristics['traffic_volume'].get('total_bytes', 0)
            if total_bytes > 1000000:  # > 1MB
                base_risk += 0.1
        
        return min(1.0, base_risk)
    
    def get_threat_trends(self, historical_data: List[Dict]) -> Dict:
        """Analyze threat trends over time"""
        
        if not historical_data:
            return {}
        
        trends = {
            'detection_rate_trend': [],
            'threat_level_distribution': {},
            'attack_type_frequency': {},
            'evasion_success_trend': []
        }
        
        for data in historical_data:
            # Detection rate trend
            trends['detection_rate_trend'].append({
                'timestamp': data.get('timestamp'),
                'detection_rate': data.get('detection_rate', 0)
            })
            
            # Threat level distribution
            threat_level = data.get('threat_level', 'UNKNOWN')
            trends['threat_level_distribution'][threat_level] = trends['threat_level_distribution'].get(threat_level, 0) + 1
            
            # Evasion success trend
            if 'evasion_analysis' in data:
                evasion_rate = data['evasion_analysis'].get('evasion_rate', 0)
                trends['evasion_success_trend'].append({
                    'timestamp': data.get('timestamp'),
                    'evasion_rate': evasion_rate
                })
        
        return trends