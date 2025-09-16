#!/usr/bin/env python3
"""
TrackPilot Override Logger and Learning System

This module handles logging of human operator overrides and provides mechanisms
for the AI system to learn from these interventions.
"""

import json
import csv
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import sqlite3
import logging

logger = logging.getLogger(__name__)


@dataclass
class OverrideEvent:
    """Data class for override events"""
    timestamp: str
    operator_id: str
    train_id: str
    section_id: str
    original_recommendation: Dict[str, Any]
    modified_schedule: Dict[str, Any]
    reason: str
    feedback: str
    confidence_before: float
    confidence_after: float
    system_metrics: Dict[str, float]
    environmental_factors: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OverrideEvent':
        """Create from dictionary"""
        return cls(**data)


class OverrideLogger:
    """Centralized logging system for human operator overrides"""
    
    def __init__(self, 
                 log_directory: str = "logs/overrides",
                 use_database: bool = True,
                 max_memory_events: int = 1000):
        """
        Initialize override logger
        
        Args:
            log_directory: Directory to store log files
            use_database: Whether to use SQLite database for persistence
            max_memory_events: Maximum events to keep in memory
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        self.use_database = use_database
        self.max_memory_events = max_memory_events
        
        # In-memory storage for recent events
        self.recent_events: deque = deque(maxlen=max_memory_events)
        
        # Statistics tracking
        self.stats = {
            'total_overrides': 0,
            'overrides_by_operator': defaultdict(int),
            'overrides_by_reason': defaultdict(int),
            'overrides_by_train': defaultdict(int),
            'avg_confidence_delta': 0.0
        }
        
        # Initialize storage backends
        self._init_file_logging()
        if self.use_database:
            self._init_database()
    
    def _init_file_logging(self):
        """Initialize file-based logging"""
        # JSON log file (detailed events)
        self.json_log_path = self.log_directory / "overrides.jsonl"
        
        # CSV log file (summary data)
        self.csv_log_path = self.log_directory / "overrides_summary.csv"
        
        # Create CSV header if file doesn't exist
        if not self.csv_log_path.exists():
            with open(self.csv_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'operator_id', 'train_id', 'section_id',
                    'reason', 'confidence_before', 'confidence_after',
                    'confidence_delta', 'feedback'
                ])
    
    def _init_database(self):
        """Initialize SQLite database for structured queries"""
        self.db_path = self.log_directory / "overrides.db"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS overrides (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    train_id TEXT NOT NULL,
                    section_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    feedback TEXT,
                    confidence_before REAL,
                    confidence_after REAL,
                    confidence_delta REAL,
                    original_recommendation TEXT,
                    modified_schedule TEXT,
                    system_metrics TEXT,
                    environmental_factors TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON overrides(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_operator ON overrides(operator_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_train ON overrides(train_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reason ON overrides(reason)")
    
    def log_override(self, override_event: OverrideEvent):
        """Log an override event to all configured backends"""
        try:
            # Add to in-memory storage
            self.recent_events.append(override_event)
            
            # Update statistics
            self._update_stats(override_event)
            
            # Log to JSON file
            self._log_to_json(override_event)
            
            # Log to CSV file
            self._log_to_csv(override_event)
            
            # Log to database
            if self.use_database:
                self._log_to_database(override_event)
            
            logger.info(f"Logged override event: {override_event.train_id} by {override_event.operator_id}")
            
        except Exception as e:
            logger.error(f"Failed to log override event: {e}")
            raise
    
    def _log_to_json(self, event: OverrideEvent):
        """Log event to JSONL file"""
        with open(self.json_log_path, 'a') as f:
            json.dump(event.to_dict(), f)
            f.write('\n')
    
    def _log_to_csv(self, event: OverrideEvent):
        """Log event summary to CSV file"""
        with open(self.csv_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            confidence_delta = event.confidence_after - event.confidence_before
            writer.writerow([
                event.timestamp,
                event.operator_id,
                event.train_id,
                event.section_id,
                event.reason,
                event.confidence_before,
                event.confidence_after,
                confidence_delta,
                event.feedback
            ])
    
    def _log_to_database(self, event: OverrideEvent):
        """Log event to SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            confidence_delta = event.confidence_after - event.confidence_before
            conn.execute("""
                INSERT INTO overrides (
                    timestamp, operator_id, train_id, section_id, reason,
                    feedback, confidence_before, confidence_after, confidence_delta,
                    original_recommendation, modified_schedule, system_metrics,
                    environmental_factors
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.timestamp,
                event.operator_id,
                event.train_id,
                event.section_id,
                event.reason,
                event.feedback,
                event.confidence_before,
                event.confidence_after,
                confidence_delta,
                json.dumps(event.original_recommendation),
                json.dumps(event.modified_schedule),
                json.dumps(event.system_metrics),
                json.dumps(event.environmental_factors)
            ))
    
    def _update_stats(self, event: OverrideEvent):
        """Update internal statistics"""
        self.stats['total_overrides'] += 1
        self.stats['overrides_by_operator'][event.operator_id] += 1
        self.stats['overrides_by_reason'][event.reason] += 1
        self.stats['overrides_by_train'][event.train_id] += 1
        
        # Update average confidence delta
        confidence_delta = event.confidence_after - event.confidence_before
        total = self.stats['total_overrides']
        current_avg = self.stats['avg_confidence_delta']
        self.stats['avg_confidence_delta'] = ((current_avg * (total - 1)) + confidence_delta) / total
    
    def get_recent_overrides(self, limit: int = 100) -> List[OverrideEvent]:
        """Get recent override events from memory"""
        return list(self.recent_events)[-limit:]
    
    def query_overrides(self, 
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None,
                       operator_id: Optional[str] = None,
                       train_id: Optional[str] = None,
                       reason: Optional[str] = None,
                       limit: int = 1000) -> List[Dict[str, Any]]:
        """Query override events with filters"""
        if not self.use_database:
            # Fallback to in-memory filtering
            results = []
            for event in self.recent_events:
                if self._matches_filters(event, start_time, end_time, 
                                       operator_id, train_id, reason):
                    results.append(event.to_dict())
                    if len(results) >= limit:
                        break
            return results
        
        # Database query
        query = "SELECT * FROM overrides WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if operator_id:
            query += " AND operator_id = ?"
            params.append(operator_id)
        
        if train_id:
            query += " AND train_id = ?"
            params.append(train_id)
        
        if reason:
            query += " AND reason = ?"
            params.append(reason)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def _matches_filters(self, event: OverrideEvent, start_time, end_time, 
                        operator_id, train_id, reason) -> bool:
        """Check if event matches query filters"""
        if start_time and event.timestamp < start_time:
            return False
        if end_time and event.timestamp > end_time:
            return False
        if operator_id and event.operator_id != operator_id:
            return False
        if train_id and event.train_id != train_id:
            return False
        if reason and event.reason != reason:
            return False
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get override statistics"""
        return dict(self.stats)
    
    def export_training_data(self, output_path: str, format: str = 'json') -> str:
        """Export override data for training purposes"""
        if not self.use_database:
            raise ValueError("Database required for data export")
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT timestamp, operator_id, train_id, section_id, reason,
                       confidence_before, confidence_after, original_recommendation,
                       modified_schedule, system_metrics, environmental_factors
                FROM overrides
                ORDER BY timestamp DESC
            """
            
            if format.lower() == 'json':
                cursor = conn.execute(query)
                data = []
                for row in cursor.fetchall():
                    record = {
                        'timestamp': row[0],
                        'operator_id': row[1],
                        'train_id': row[2],
                        'section_id': row[3],
                        'reason': row[4],
                        'confidence_before': row[5],
                        'confidence_after': row[6],
                        'original_recommendation': json.loads(row[7]) if row[7] else {},
                        'modified_schedule': json.loads(row[8]) if row[8] else {},
                        'system_metrics': json.loads(row[9]) if row[9] else {},
                        'environmental_factors': json.loads(row[10]) if row[10] else {}
                    }
                    data.append(record)
                
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format.lower() == 'csv':
                df = pd.read_sql_query(query, conn)
                df.to_csv(output_path, index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported override data to {output_path}")
        return output_path


class OverrideLearningSystem:
    """System for learning from human operator overrides"""
    
    def __init__(self, override_logger: OverrideLogger):
        self.logger = override_logger
        self.learning_enabled = True
        self.min_overrides_for_learning = 10
    
    def analyze_override_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in override behavior"""
        recent_overrides = self.logger.get_recent_overrides(1000)
        
        if len(recent_overrides) < self.min_overrides_for_learning:
            return {"status": "insufficient_data", "count": len(recent_overrides)}
        
        # Convert to DataFrame for analysis
        data = []
        for event in recent_overrides:
            data.append({
                'operator_id': event.operator_id,
                'reason': event.reason,
                'confidence_before': event.confidence_before,
                'confidence_after': event.confidence_after,
                'confidence_delta': event.confidence_after - event.confidence_before,
                'hour_of_day': datetime.fromisoformat(event.timestamp).hour,
                'day_of_week': datetime.fromisoformat(event.timestamp).weekday()
            })
        
        df = pd.DataFrame(data)
        
        analysis = {
            'total_overrides': len(df),
            'operators': {
                'count': df['operator_id'].nunique(),
                'most_active': df['operator_id'].value_counts().head().to_dict()
            },
            'reasons': {
                'distribution': df['reason'].value_counts().to_dict(),
                'by_confidence_impact': df.groupby('reason')['confidence_delta'].mean().to_dict()
            },
            'temporal_patterns': {
                'by_hour': df.groupby('hour_of_day')['confidence_delta'].mean().to_dict(),
                'by_day': df.groupby('day_of_week')['confidence_delta'].mean().to_dict()
            },
            'confidence_analysis': {
                'avg_before': float(df['confidence_before'].mean()),
                'avg_after': float(df['confidence_after'].mean()),
                'avg_delta': float(df['confidence_delta'].mean()),
                'improvement_rate': float((df['confidence_delta'] > 0).mean())
            }
        }
        
        return analysis
    
    def generate_learning_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations for model improvement based on overrides"""
        patterns = self.analyze_override_patterns()
        
        if patterns.get('status') == 'insufficient_data':
            return []
        
        recommendations = []
        
        # Check for frequent override reasons
        reason_counts = patterns['reasons']['distribution']
        total_overrides = patterns['total_overrides']
        
        for reason, count in reason_counts.items():
            if count / total_overrides > 0.2:  # More than 20% of overrides
                recommendations.append({
                    'type': 'frequent_override_reason',
                    'reason': reason,
                    'frequency': count / total_overrides,
                    'suggestion': f'Consider improving model handling of {reason} scenarios',
                    'priority': 'high'
                })
        
        # Check for temporal patterns
        temporal = patterns['temporal_patterns']
        
        # Find hours with frequent overrides
        hour_overrides = [(hour, abs(delta)) for hour, delta in temporal['by_hour'].items()]
        hour_overrides.sort(key=lambda x: x[1], reverse=True)
        
        if hour_overrides and hour_overrides[0][1] > 0.1:  # Significant confidence impact
            recommendations.append({
                'type': 'temporal_pattern',
                'pattern': 'hourly_bias',
                'peak_hours': [h for h, d in hour_overrides[:3]],
                'suggestion': 'Model may have time-of-day bias requiring attention',
                'priority': 'medium'
            })
        
        # Check operator consistency
        operators = patterns['operators']
        if operators['count'] > 1:
            most_active = max(operators['most_active'].values())
            if most_active / total_overrides > 0.5:  # One operator does >50% of overrides
                recommendations.append({
                    'type': 'operator_concentration',
                    'suggestion': 'Consider training more operators or investigating model acceptance',
                    'priority': 'medium'
                })
        
        # Check confidence improvement rate
        confidence = patterns['confidence_analysis']
        if confidence['improvement_rate'] < 0.3:  # Less than 30% of overrides improve confidence
            recommendations.append({
                'type': 'low_improvement_rate',
                'current_rate': confidence['improvement_rate'],
                'suggestion': 'Many overrides do not improve confidence - review override criteria',
                'priority': 'high'
            })
        
        return recommendations
    
    def prepare_training_data(self, output_dir: str) -> str:
        """Prepare override data for model retraining"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export raw data
        raw_data_path = os.path.join(output_dir, 'override_data.json')
        self.logger.export_training_data(raw_data_path, 'json')
        
        # Create training-specific format
        training_data = []
        recent_overrides = self.logger.get_recent_overrides(1000)
        
        for event in recent_overrides:
            # Convert to format suitable for behavioral cloning
            training_record = {
                'state': self._extract_state_features(event),
                'action': self._extract_action_label(event),
                'reward': 1.0 if event.confidence_after > event.confidence_before else 0.0,
                'metadata': {
                    'timestamp': event.timestamp,
                    'operator_id': event.operator_id,
                    'reason': event.reason,
                    'confidence_delta': event.confidence_after - event.confidence_before
                }
            }
            training_data.append(training_record)
        
        # Save processed training data
        training_path = os.path.join(output_dir, 'processed_training_data.json')
        with open(training_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Prepared {len(training_data)} training examples in {training_path}")
        return training_path
    
    def _extract_state_features(self, event: OverrideEvent) -> List[float]:
        """Extract state features for training"""
        # This should match the state representation used by the model
        features = []
        
        # System metrics
        metrics = event.system_metrics
        features.extend([
            metrics.get('traffic_density', 0.5),
            metrics.get('avg_delay', 0.0),
            metrics.get('section_utilization', 0.5),
            event.confidence_before
        ])
        
        # Environmental factors
        env_factors = event.environmental_factors
        features.extend([
            env_factors.get('weather_score', 0.5),
            env_factors.get('time_of_day', 12) / 24.0,
            env_factors.get('day_of_week', 3) / 7.0
        ])
        
        # Pad to fixed size if necessary
        while len(features) < 64:  # Assuming 64-dimensional state
            features.append(0.0)
        
        return features[:64]  # Ensure exact size
    
    def _extract_action_label(self, event: OverrideEvent) -> int:
        """Extract action label from override"""
        # Convert override reason to action class
        reason_to_action = {
            'delay_train': 1,
            'priority_override': 2,
            'route_change': 3,
            'emergency_stop': 4,
            'safety_concern': 4,
            'efficiency_improvement': 2,
            'passenger_priority': 2,
            'maintenance_window': 3
        }
        
        return reason_to_action.get(event.reason, 0)  # Default to 'maintain' action
    
    def set_learning_enabled(self, enabled: bool):
        """Enable or disable learning from overrides"""
        self.learning_enabled = enabled
        logger.info(f"Override learning {'enabled' if enabled else 'disabled'}")


def create_sample_override(operator_id: str = "operator_001") -> OverrideEvent:
    """Create a sample override event for testing"""
    return OverrideEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        operator_id=operator_id,
        train_id="TRAIN_12345",
        section_id="SECTION_A1",
        original_recommendation={
            "entry_time": "2024-01-15T14:30:00Z",
            "exit_time": "2024-01-15T14:45:00Z",
            "confidence": 0.85
        },
        modified_schedule={
            "entry_time": "2024-01-15T14:35:00Z",
            "exit_time": "2024-01-15T14:50:00Z",
            "confidence": 0.92
        },
        reason="safety_concern",
        feedback="Adjusted timing due to observed track maintenance activity",
        confidence_before=0.85,
        confidence_after=0.92,
        system_metrics={
            "traffic_density": 0.7,
            "avg_delay": 5.2,
            "section_utilization": 0.8
        },
        environmental_factors={
            "weather_score": 0.9,
            "time_of_day": 14,
            "day_of_week": 1
        }
    )


if __name__ == "__main__":
    # Example usage
    logger = OverrideLogger("logs/overrides")
    learning_system = OverrideLearningSystem(logger)
    
    # Log some sample overrides
    for i in range(5):
        sample_override = create_sample_override(f"operator_{i+1:03d}")
        logger.log_override(sample_override)
    
    # Analyze patterns
    patterns = learning_system.analyze_override_patterns()
    print("Override Patterns:", json.dumps(patterns, indent=2))
    
    # Generate recommendations
    recommendations = learning_system.generate_learning_recommendations()
    print("Learning Recommendations:", json.dumps(recommendations, indent=2))
    
    # Prepare training data
    training_path = learning_system.prepare_training_data("training_output")
    print(f"Training data prepared: {training_path}")