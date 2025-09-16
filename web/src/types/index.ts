// TrackPilot Frontend Type Definitions

export interface TrainInfo {
  train_id: string;
  route: string;
  scheduled_arrival: string;
  scheduled_departure: string;
  priority: number; // 1-10
  speed_kmh: number;
  is_freight: boolean;
  stops: string[];
  current_position?: {
    section_id: string;
    position_km: number;
  };
}

export interface SectionInfo {
  section_id: string;
  length_km: number;
  max_trains: number;
  is_single_track: boolean;
  signals: string[];
  current_occupancy: number;
}

export interface ScheduleEntry {
  train_id: string;
  section_id: string;
  entry_time: string;
  exit_time: string;
  confidence_score: number;
  explanation: string;
}

export interface AIRecommendation {
  schedule: ScheduleEntry[];
  overall_confidence: number;
  reasoning: string;
  metrics: Record<string, number>;
  timestamp: string;
}

export interface HumanOverride {
  timestamp: string;
  operator_id: string;
  original_recommendation: AIRecommendation;
  modified_schedule: ScheduleEntry[];
  reason: string;
  feedback: string;
}

export interface RuleViolation {
  rule_id: string;
  train_id: string;
  section_id: string;
  description: string;
  severity: number; // 0.0-1.0
}

export interface SystemMetrics {
  total_trains: number;
  active_sections: number;
  avg_confidence: number;
  override_rate: number;
  efficiency_score: number;
  safety_score: number;
  throughput: number;
  avg_delay_minutes: number;
  last_updated: string;
}

// UI State Types
export interface ViewportTransform {
  x: number;
  y: number;
  k: number; // zoom scale
}

export interface TimeRange {
  start: Date;
  end: Date;
}

export interface FilterState {
  trains: string[];
  sections: string[];
  priorities: number[];
  showFreight: boolean;
  showPassenger: boolean;
  timeRange: TimeRange;
}

// D3 Visualization Types
export interface SpaceTimePoint {
  train_id: string;
  section_id: string;
  time: Date;
  position: number; // 0-1 normalized position in section
  speed_kmh: number;
  status: 'scheduled' | 'actual' | 'predicted';
}

export interface TrainPath {
  train_id: string;
  points: SpaceTimePoint[];
  color: string;
  priority: number;
  is_freight: boolean;
}

export interface ConflictArea {
  section_id: string;
  time_start: Date;
  time_end: Date;
  trains: string[];
  severity: 'low' | 'medium' | 'high';
}

// API Response Types
export interface APIResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

export interface ScheduleRequest {
  trains: TrainInfo[];
  sections: SectionInfo[];
  horizon_hours: number;
  constraints?: Record<string, any>;
}

export interface OverrideRequest {
  override_id: string;
  modified_schedule: ScheduleEntry[];
  reason: string;
  feedback?: string;
  operator_id: string;
}

// Real-time Update Types
export interface RealTimeUpdate {
  type: 'train_position' | 'schedule_change' | 'alert' | 'override';
  timestamp: string;
  data: any;
}

export interface TrainPositionUpdate extends RealTimeUpdate {
  type: 'train_position';
  data: {
    train_id: string;
    section_id: string;
    position_km: number;
    speed_kmh: number;
    delay_minutes: number;
  };
}

export interface ScheduleChangeUpdate extends RealTimeUpdate {
  type: 'schedule_change';
  data: {
    affected_trains: string[];
    new_schedule: ScheduleEntry[];
    reason: string;
  };
}

export interface AlertUpdate extends RealTimeUpdate {
  type: 'alert';
  data: {
    level: 'info' | 'warning' | 'error' | 'critical';
    title: string;
    message: string;
    train_id?: string;
    section_id?: string;
    requires_action: boolean;
  };
}

// Component Props Types
export interface SpaceTimeGraphProps {
  data: TrainPath[];
  sections: SectionInfo[];
  timeRange: TimeRange;
  width: number;
  height: number;
  onTrainClick?: (trainId: string) => void;
  onTimeRangeChange?: (range: TimeRange) => void;
  conflicts?: ConflictArea[];
}

export interface ControlPanelProps {
  recommendation: AIRecommendation | null;
  onOverride: (override: OverrideRequest) => void;
  onAccept: () => void;
  onReject: () => void;
  isLoading: boolean;
}

export interface TrainListProps {
  trains: TrainInfo[];
  selectedTrain?: string;
  onTrainSelect: (trainId: string) => void;
  onTrainUpdate: (train: TrainInfo) => void;
}

export interface MetricsDashboardProps {
  metrics: SystemMetrics;
  historicalData?: SystemMetrics[];
  refreshInterval?: number;
}

// Store State Types
export interface AppState {
  // Data
  trains: TrainInfo[];
  sections: SectionInfo[];
  currentRecommendation: AIRecommendation | null;
  systemMetrics: SystemMetrics | null;
  
  // UI State
  selectedTrain: string | null;
  filter: FilterState;
  viewport: ViewportTransform;
  isLoading: boolean;
  error: string | null;
  
  // Real-time
  isConnected: boolean;
  lastUpdate: string | null;
  
  // Actions
  setTrains: (trains: TrainInfo[]) => void;
  setSections: (sections: SectionInfo[]) => void;
  setRecommendation: (recommendation: AIRecommendation) => void;
  setMetrics: (metrics: SystemMetrics) => void;
  setSelectedTrain: (trainId: string | null) => void;
  setFilter: (filter: Partial<FilterState>) => void;
  setViewport: (transform: ViewportTransform) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setConnectionStatus: (connected: boolean) => void;
}

// Configuration Types
export interface AppConfig {
  apiUrl: string;
  wsUrl: string;
  refreshInterval: number;
  maxTrainsDisplayed: number;
  defaultTimeHorizon: number;
  theme: 'light' | 'dark';
  enableSounds: boolean;
  enableNotifications: boolean;
}

// Utility Types
export type TrainStatus = 'scheduled' | 'delayed' | 'on_time' | 'cancelled' | 'diverted';
export type LogLevel = 'debug' | 'info' | 'warning' | 'error' | 'critical';
export type OperatorRole = 'viewer' | 'operator' | 'supervisor' | 'admin';

export interface UserInfo {
  operator_id: string;
  name: string;
  role: OperatorRole;
  permissions: string[];
  last_login: string;
}

// Event Handler Types
export type TrainEventHandler = (train: TrainInfo) => void;
export type OverrideEventHandler = (override: HumanOverride) => void;
export type AlertEventHandler = (alert: AlertUpdate['data']) => void;
export type ErrorEventHandler = (error: Error) => void;